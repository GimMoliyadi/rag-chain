"""全局配置。

这个模块只做一件事：把 .env 里的参数读进来，交给其它模块使用。
改行为只需要改 .env，不用碰代码。
"""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录 = app/ 的上一级，也就是 requirements.txt 所在的那一层
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# .env 里允许填的合法值，集中放这里，validate_providers 只做比对
EMBEDDING_PROVIDERS = {"local", "local_semantic", "openai"}
LLM_PROVIDERS = {"mock", "openai", "anthropic"}
LOCAL_DEVICES = {"auto", "cpu", "cuda", "mps"}
LOCAL_DTYPES = {"auto", "fp16", "fp32"}

# 单条文本的 token 上限默认值。
# 实测 14303 个 chunk 的 token 长度：p50=304、p99=780、max=3654。
# 1024 只截断 12 条（0.08%）的尾部，却把最坏情况的显存锁死。
DEFAULT_MAX_LENGTH = 1024

# 默认精排模型。中英双语，和 bge-m3 同家族（共享 XLM-RoBERTa 底座）。
# 选 v2-m3 而不是 v1：v2 在 MIRACL 多语言榜上重训过，中文更好；
# 选 m3 而不是更大的模型：fp16 约 1.12 GB，6GB 显存装得下。
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


class Settings(BaseSettings):
    """集中管理所有可调参数。

    pydantic-settings 的取值优先级（高 -> 低）：
    构造时传参 > 环境变量 > .env 文件 > 这里写的默认值。
    测试里传构造参数就能把数据库和向量库重定向到临时目录，不污染真实数据。
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",  # .env 里多出来的字段不报错，方便加临时调试项
    )

    # ---- 知识库路径（只读！）----
    vault_path: str = str(Path.home() / "Documents" / "Obsidian Vault")

    # ---- 数据输出路径，相对路径会拼到项目根目录下 ----
    db_path: str = "data/knowledge.db"
    chroma_path: str = "data/chroma"

    # ---- 切块与检索 ----
    chunk_size: int = 800       # 单个 chunk 的目标中文字符数
    chunk_overlap: int = 120    # 相邻 chunk 的重叠字符数
    top_k: int = 5              # 检索默认返回条数

    # ---- Context 清洗（retrieval 和 LLM 之间的一层）----
    # 默认关闭：评估阶段要随时拿回"原始检索结果"的数字当 baseline，
    # 开关一开一关必须能复现同一个数字。
    # 开启后流程变成：取 Top-20 -> 过滤导航 chunk -> 取 Top-K ->
    # 清理 JSON 导出尾巴 -> 拼 context。SQLite 里的 chunk 原文始终不动，
    # sources 仍然引用原始 chunk。
    context_filter_enabled: bool = False
    filter_retrieve_top_k: int = 20   # 开启过滤时先多取多少条候选

    # ---- Reranker（精排，实验性）----
    # 默认关闭有两个原因：评估阶段要随时拿回原始检索结果当 baseline（开关一开
    # 一关必须复现同一个数字），以及加载模型要占约 1.1 GB 显存，没开就不该碰。
    # 开启后流程变成：检索 Top-N -> cross-encoder 打分重排 -> 过滤 -> Top-K。
    # N 取 50 而不是和 filter 一样的 20，依据是 scripts/probe_reranker_headroom.py
    # 在 Top-50 下量出的期望文档排名是 #7 ~ #33：用 20 会把 #24 和 #33 那两篇
    # 直接排除在候选池外，精排彻底救不了它们。这个数字是量出来的，不是拍的。
    reranker_enabled: bool = False
    reranker_model: str = DEFAULT_RERANKER_MODEL
    reranker_candidate_top_k: int = 50
    reranker_device: str = "auto"
    reranker_dtype: str = "auto"
    reranker_batch_size: int = 8

    # ---- Embedding ----
    # local          = 哈希伪向量（测试 fallback，无语义理解）
    # local_semantic = 本地 sentence-transformers 模型（真语义，离线，推荐）
    # openai         = OpenAI 兼容接口
    embedding_provider: str = "local"
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    embedding_model: str = ""
    embedding_dim: int = 256    # 仅 local 模式使用

    # ---- 本地语义 Embedding（仅 local_semantic 模式生效）----
    local_embedding_model: str = "BAAI/bge-m3"   # 模型名，改这里就能换模型，不用改代码
    local_embedding_device: str = "auto"          # auto = 有 CUDA 用 CUDA，否则 CPU
    local_embedding_query_prefix: str = ""        # 需要指令前缀的模型在这里填，默认不加
    local_embedding_max_length: int = DEFAULT_MAX_LENGTH  # 单条 token 上限，超出截断
    local_embedding_dtype: str = "auto"           # auto = CUDA 用 fp16，CPU 用 fp32
    # 单次调用带的 chunk 数。8 是实测值，不是猜的：
    # RTX 4050 6GB + fp16 + max_seq_length=1024，用 16 条接近 1024 token 的真实 chunk 压测
    # （scripts/probe_batch_memory.py），batch=8 峰值 reserved 1354 MiB、25.8 条/s；
    # batch=16 只多 256 MiB 却只换来 26.7 条/s（+3.5%）。宁可慢也不要重建中途 OOM。
    embedding_batch_size: int = 8

    # ---- 知识库显示名，会出现在 RAG 的 system prompt 里 ----
    vault_name: str = "Obsidian"

    # ---- LLM ----
    # mock      = 离线回显（跑通链路用）
    # openai    = OpenAI 兼容 /chat/completions
    # anthropic = Anthropic Messages API；模型/端点/凭证三处取值见下面的 anthropic_* 属性
    llm_provider: str = "mock"
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    # Anthropic Messages API 里 max_tokens 是必填项。4096 的依据：本知识库单轮回答
    # 实测几百 token，留足长引用回答的余量；不给更大的值，因为代理按 max_tokens
    # 预留额度而不是按实际输出预留，虚高只会白占配额。
    llm_max_tokens: int = 4096
    # anthropic SDK 默认 Timeout(timeout=600, connect=5.0)，两个值都刻意覆盖，
    # 但管的是同一次请求的不同阶段：connect 只管"连不连得上"，给连不上时一个
    # 快速失败的上限；total 才是等回答的耗时，必须给长 context 留余量。
    #
    # 上游 LLM 延迟受服务负载、网络和时段影响较大。本项目的历史实测在不同时间
    # 给出过 8 秒级、数十秒、更长的结果，已经证明不稳定，所以这里不再记录任何
    # 单次测量值。需要新数字时跑 `python scripts/probe_llm_latency.py`，
    # 并在多个时段重复测量后再决定要不要改这两个值。
    # 两个值都保持可配置（.env 的 LLM_TIMEOUT_CONNECT / LLM_TIMEOUT_TOTAL）。
    llm_timeout_connect: float = 15.0
    llm_timeout_total: float = 300.0

    @property
    def db_path_resolved(self) -> Path:
        """把 db_path 统一解析成绝对路径。"""
        return _resolve_path(self.db_path)

    @property
    def chroma_path_resolved(self) -> Path:
        return _resolve_path(self.chroma_path)

    @property
    def vault_path_resolved(self) -> Path:
        return Path(self.vault_path).expanduser().resolve()

    # anthropic 的模型 / 凭证 / 端点三处取值用同一条规则：
    # .env 显式填的值优先，进程环境的 ANTHROPIC_* 兜底，空白视为没填。
    # 选这个顺序是因为 .env 是本项目自己定的配置，而 ANTHROPIC_MODEL 反映的是
    # 「当前 Claude Code 会话」正在用的模型——让环境变量静默覆盖会让项目配置失效。
    # 三处共用一条规则，省得优先级逻辑在三个地方各写一遍、以后改一处漏两处。
    @property
    def anthropic_model(self) -> str:
        """实际生效的模型 ID：.env 的 LLM_MODEL 优先，其次进程环境的 ANTHROPIC_MODEL。"""
        return _first_nonempty(self.llm_model, "ANTHROPIC_MODEL")

    @property
    def anthropic_api_key(self) -> str:
        """实际生效的凭证：.env 的 LLM_API_KEY 优先，其次进程环境的 ANTHROPIC_AUTH_TOKEN。"""
        return _first_nonempty(self.llm_api_key, "ANTHROPIC_AUTH_TOKEN")

    @property
    def anthropic_base_url(self) -> str:
        """实际生效的端点：.env 的 LLM_BASE_URL 优先，其次进程环境的 ANTHROPIC_BASE_URL。"""
        return _first_nonempty(self.llm_base_url, "ANTHROPIC_BASE_URL")

    def validate_providers(self) -> None:
        """启动前校验配置是否自洽，配置错了就早失败，别等到请求时才炸。"""
        if self.embedding_provider not in EMBEDDING_PROVIDERS:
            raise ValueError(
                f"EMBEDDING_PROVIDER 只能是 {'/'.join(sorted(EMBEDDING_PROVIDERS))}，"
                f"当前：{self.embedding_provider!r}"
            )
        if self.llm_provider not in LLM_PROVIDERS:
            raise ValueError(
                f"LLM_PROVIDER 只能是 {'/'.join(sorted(LLM_PROVIDERS))}，"
                f"当前：{self.llm_provider!r}"
            )
        if self.embedding_provider == "openai" and not (self.embedding_base_url and self.embedding_api_key):
            raise ValueError("EMBEDDING_PROVIDER=openai 时必须同时配置 EMBEDDING_BASE_URL 和 EMBEDDING_API_KEY")
        if self.llm_provider == "openai" and not (self.llm_base_url and self.llm_api_key):
            raise ValueError("LLM_PROVIDER=openai 时必须同时配置 LLM_BASE_URL 和 LLM_API_KEY")
        # anthropic 的模型 / 凭证 / 端点都能从 .env 或进程环境两处取（见上面的 anthropic_* 属性），
        # 所以这里只校验「够不够用」。token 的值本身不进日志、不打印，
        # 报错文案只提变量名，方便照名字排查。
        if self.llm_provider == "anthropic":
            if not self.anthropic_api_key:
                raise ValueError(
                    "LLM_PROVIDER=anthropic 但凭证为空：.env 的 LLM_API_KEY 和进程环境的 "
                    "ANTHROPIC_AUTH_TOKEN 都没有。请在其中一处填上。"
                )
            # 模型 ID 两处都可能填，两处都空时 SDK 会以"缺少 model 参数"的方式失败，
            # 那个报错看不出是配置问题，不如在这里说清楚。
            if not self.anthropic_model:
                raise ValueError(
                    "LLM_PROVIDER=anthropic 但模型 ID 为空：.env 的 LLM_MODEL 和进程环境的 "
                    "ANTHROPIC_MODEL 都没有。请在其中一处填上。"
                )
            # BASE_URL 可以留空（SDK 默认走 api.anthropic.com），填了就校验格式。
            # 最常见的错是漏掉 http(s):// 前缀，那种情况下 SDK 抛的错非常绕。
            base_url = self.anthropic_base_url
            if base_url:
                parsed = urlparse(base_url)
                if parsed.scheme not in ("http", "https") or not parsed.netloc:
                    raise ValueError(
                        "ANTHROPIC_BASE_URL 不是合法的 http(s) 地址，"
                        "常见的错法是漏了 http:// 或 https:// 前缀。"
                    )
        if self.local_embedding_device not in LOCAL_DEVICES:
            raise ValueError(
                f"LOCAL_EMBEDDING_DEVICE 只能是 {'/'.join(sorted(LOCAL_DEVICES))}，"
                f"当前：{self.local_embedding_device!r}"
            )
        if self.local_embedding_dtype not in LOCAL_DTYPES:
            raise ValueError(
                f"LOCAL_EMBEDDING_DTYPE 只能是 {'/'.join(sorted(LOCAL_DTYPES))}，"
                f"当前：{self.local_embedding_dtype!r}"
            )
        if self.local_embedding_max_length < 1:
            raise ValueError(
                f"LOCAL_EMBEDDING_MAX_LENGTH 必须是正整数，当前：{self.local_embedding_max_length}"
            )
        if self.embedding_batch_size < 1:
            raise ValueError(f"EMBEDDING_BATCH_SIZE 必须是正整数，当前：{self.embedding_batch_size}")
        # 精排只在显式开启时才有意义，所以校验也只在开启时触发；
        # 关着的时候填了错的设备名不该让服务起不来。
        if self.reranker_enabled:
            if self.reranker_device not in LOCAL_DEVICES:
                raise ValueError(
                    f"RERANKER_DEVICE 只能是 {'/'.join(sorted(LOCAL_DEVICES))}，"
                    f"当前：{self.reranker_device!r}"
                )
            if self.reranker_dtype not in LOCAL_DTYPES:
                raise ValueError(
                    f"RERANKER_DTYPE 只能是 {'/'.join(sorted(LOCAL_DTYPES))}，"
                    f"当前：{self.reranker_dtype!r}"
                )
            if self.reranker_candidate_top_k < 1:
                raise ValueError(
                    f"RERANKER_CANDIDATE_TOP_K 必须是正整数，当前：{self.reranker_candidate_top_k}"
                )
            if self.reranker_batch_size < 1:
                raise ValueError(
                    f"RERANKER_BATCH_SIZE 必须是正整数，当前：{self.reranker_batch_size}"
                )


def _first_nonempty(value: str, env_name: str) -> str:
    """取值优先级：显式配置优先，同名进程环境变量兜底。空白字符视为没填。"""
    return value.strip() or os.environ.get(env_name, "").strip()


def _resolve_path(path: str) -> Path:
    """相对路径相对项目根目录解析，绝对路径原样返回。"""
    p = Path(path).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p
