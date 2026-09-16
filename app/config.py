"""全局配置。

这个模块只做一件事：把 .env 里的参数读进来，交给其它模块使用。
改行为只需要改 .env，不用碰代码。
"""
from __future__ import annotations

import os
from pathlib import Path

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
    # anthropic = Anthropic Messages API；凭证不经过 .env，SDK 直接读进程里的 ANTHROPIC_*
    llm_provider: str = "mock"
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    # Anthropic Messages API 里 max_tokens 是必填项。4096 的依据：本知识库单轮回答
    # 实测几百 token，留足长引用回答的余量；不给更大的值，因为代理按 max_tokens
    # 预留额度而不是按实际输出预留，虚高只会白占配额。
    llm_max_tokens: int = 4096

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
        # anthropic 的凭证刻意不落 .env：SDK 从进程环境的 ANTHROPIC_AUTH_TOKEN 取，
        # 这样 token 既不进代码也不写文件，也能直接复用 Claude Code 的进程环境。
        # 这里只检查「存不存在」，值本身不读、不打印。
        if self.llm_provider == "anthropic" and not os.environ.get("ANTHROPIC_AUTH_TOKEN", "").strip():
            raise ValueError(
                "LLM_PROVIDER=anthropic 但进程环境里没有 ANTHROPIC_AUTH_TOKEN。"
                "这个实现复用 Claude Code 的 ANTHROPIC_* 变量，不要把 token 写进 .env。"
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


def _resolve_path(path: str) -> Path:
    """相对路径相对项目根目录解析，绝对路径原样返回。"""
    p = Path(path).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p
