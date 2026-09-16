"""Embedding 客户端，设计成可替换的。

三个实现（最终结构）：

    BaseEmbedding
        ├── HashingEmbedding            local          哈希伪向量，测试 fallback，无语义理解
        ├── LocalSemanticEmbedding      local_semantic 本地 sentence-transformers，真语义
        └── OpenAICompatibleEmbedding   openai         任意 OpenAI 兼容的 /embeddings 接口

对外只有两个方法：
    embed(texts)      给文档 / chunk 做向量，批量输入
    embed_query(text) 给单个问题做向量

为什么 embed 和 embed_query 要分开？
    有些模型官方要求只在查询前加指令、文档前不加（例如 BAAI/bge-large-zh-v1.5）。
    把两个方法分开之后，"文档怎么编码"和"查询怎么编码"的差异只出现在需要它的那
    一个实现里。默认实现两者等价，所以 HashingEmbedding 和 OpenAICompatibleEmbedding
    不需要任何改动。

为什么不直接上 LangChain：这里每一层的输入输出都要肉眼可见，抽象掉就看不出
"14303 个 chunk 是怎么一批一批变成向量的"。
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

import numpy as np

from .config import Settings

# 中文按单字 + 相邻双字切，英文数字按连续片段切。
# 这是"特征哈希"的粒度：太粗丢信息，太细桶数爆炸。
_TOKEN_RE = re.compile(r"[一-鿿]{1,2}|[A-Za-z0-9_]+")

# 本地模型缓存：同一个进程里，同一个 (模型名, 设备, 精度, 截断长度) 组合只加载一次。
# 加载 bge-m3 要 2.17 GB 显存/内存和几十秒；如果按 chunk 反复加载，
# 14303 个 chunk 会慢到没法用。这是"模型只在程序开始加载一次"的实现。
# 截断长度也进键，原因见 _get_local_model。
_MODEL_CACHE: dict[tuple[str, str, str, int], Any] = {}


class BaseEmbedding:
    """所有 embedding 实现的共同接口。子类只需实现 embed()。"""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """把一批文本变成一批向量。输入输出长度必须一致。"""
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        """把单个问题变成向量。

        默认实现直接复用 embed()。需要区分查询和文档处理方式的模型
        （比如要加指令前缀的 BGE-zh 系列）覆写这个方法即可。
        """
        return self.embed([text])[0]

    @property
    def model_name(self) -> str:
        """模型的标识，用于日志和判断向量库是不是同一个模型建的。"""
        raise NotImplementedError

    @property
    def dimensions(self) -> int:
        raise NotImplementedError

    @property
    def device_name(self) -> str:
        """运行位置，用于启动日志。默认按本地 CPU 算。"""
        return "CPU"

    @property
    def max_length(self) -> int | None:
        """输入 token 上限；模型没有序列长度概念时返回 None。

        哈希向量按词切分、远程 API 由服务端处理，都不存在"我们这侧截断"的问题。
        返回 None 让调用方知道不用检查，而不是猜一个不存在的数字。
        """
        return None

    def count_tokens(self, texts: list[str]) -> list[int] | None:
        """返回每条文本的 token 数；无法统计时返回 None。

        只有知道 token 数才能判断一条 chunk 会不会被截尾。截尾是静默丢内容，
        所以 ingest 用这个主动报告，而不是等用户发现检索结果缺一段才回头查。
        """
        return None

    @property
    def fingerprint(self) -> dict:
        """模型身份指纹：判断向量库是不是"我"建的，只比这两个字段。

        故意不放 provider 名——model_name 本身（hashing-256d vs BAAI/bge-m3）
        已经能区分实现了，再加字段只会让两边的判断逻辑变复杂。
        """
        return {"model": self.model_name, "dimensions": self.dimensions}


class HashingEmbedding(BaseEmbedding):
    """把文本通过特征哈希映射成固定长度向量（纯离线，没有任何语义理解）。

    原理：每个 token 用 md5 决定落在哪个桶、取正还是负，最后累加并做 L2 归一化。
    相同 token 序列会稳定映射到同一桶，所以相同文本永远得到相同向量。

    它的语义能力约等于"数相同单词的个数"。"POVM" 和 "正算符值测度" 没有共同
    token，向量距离几乎不相关——这是测试 fallback 不是"简易版真实 embedding"。

    它依然有用：跑测试不需要下载模型，也不需要联网。
    """

    def __init__(self, dimensions: int = 256) -> None:
        self._dim = dimensions

    @property
    def model_name(self) -> str:
        return f"hashing-{self._dim}d"

    @property
    def dimensions(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vec = np.zeros(self._dim, dtype=np.float32)
        for token in _TOKEN_RE.findall(text.lower()):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self._dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[idx] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec.tolist()


class LocalSemanticEmbedding(BaseEmbedding):
    """本地语义向量：sentence-transformers 加载的开源模型，完全离线。

    和 HashingEmbedding 的本质区别是这里做的是"语义理解"：
    "POVM" 和 "正算符值测度" 会落在向量空间的近处，哪怕一个字符都不重合。
    哈希伪向量做不到这一点——它只会把两个字符串的公共字符映射到同一个桶。

    模型在 __init__ 里加载一次（见 _MODEL_CACHE），之后所有调用复用同一个实例。
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        batch_size: int = 8,
        query_prefix: str = "",
        max_length: int = 1024,
        dtype: str = "auto",
    ) -> None:
        self._model_name = model_name
        self._device = _resolve_device(device)
        self._dtype = _resolve_dtype(self._device, dtype)
        self._batch_size = batch_size
        self._query_prefix = query_prefix
        self._max_length = max_length
        self._model = _get_local_model(model_name, self._device, self._dtype, max_length)
        self._dim: int | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def device_name(self) -> str:
        return self._device

    @property
    def dtype_name(self) -> str:
        return self._dtype

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def fingerprint(self) -> dict:
        """指纹 = 一切会改变输出向量的东西。

        基类只放 model + dimensions。本地语义模型还有两个旋钮会改变向量：
        dtype（fp16 和 fp32 算出来的浮点数不一样）和 max_length（超长文本被
        截尾后向量就变了）。漏掉任何一个，改完 .env 后旧索引会悄悄继续被查询，
        正好是检索层最该拦住的那种事故。
        """
        return {
            "model": self._model_name,
            "dimensions": self.dimensions,
            "dtype": self._dtype,
            "max_length": self._max_length,
        }

    @property
    def dimensions(self) -> int:
        """维度不猜：第一次真正编码时探测出来并记住。"""
        if self._dim is None:
            self._dim = len(self.embed(["维度探测"])[0])
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        """给文档 / chunk 做向量，批量处理。

        不加任何前缀：这是文档侧的编码方式。BAAI/bge-m3 是专用检索模型，
        官方用法里文档和查询都不加指令前缀。需要前缀的模型
        （BAAI/bge-large-zh-v1.5 系列只要求查询加）通过 embed_query 处理，
        不要在这里污染文档侧。

        分批也在这里做：batch_size 控制一次喂给模型的文本条数，
        管的是 GPU 显存上限。调用方可以整批发来，不用自己切。
        """
        if not texts:
            return []
        vectors = [
            self._encode(texts[start : start + self._batch_size])
            for start in range(0, len(texts), self._batch_size)
        ]
        stacked = np.vstack(vectors)
        self._dim = int(stacked.shape[1])
        return stacked.tolist()

    def embed_query(self, text: str) -> list[float]:
        """给单个问题做向量。

        query_prefix 为空时（bge-m3 的官方推荐做法）和 embed() 完全等价。
        换成需要区分查询/文档的模型时，在 .env 的 LOCAL_EMBEDDING_QUERY_PREFIX
        里填官方 README 要求的那句指令，只有查询会带上它，文档侧不受影响。
        """
        query = f"{self._query_prefix}{text}" if self._query_prefix else text
        return self._encode(query).tolist()

    def count_tokens(self, texts: list[str]) -> list[int]:
        """用模型自己的 tokenizer 数 token，用来判断一条 chunk 会不会被截尾。

        显式传 truncation=False 很关键：_get_local_model 把 max_seq_length 写进了
        tokenizer，如果不禁掉截断，数出来的永远是 min(真实长度, 上限)，
        1085 token 的 chunk 会报告成 1024，一条截断都发现不了。

        真正的截断发生在 encode() 内部——库默认 truncation="longest_first"，
        这里只负责测量，不做任何改写。
        """
        encoded = self._model.tokenizer(texts, truncation=False)
        return [len(ids) for ids in encoded["input_ids"]]

    def _encode(self, texts: str | list[str]) -> np.ndarray:
        """调一次模型，不做分批——分批是 embed() 的职责。

        normalize_embeddings=True 是 BGE 官方推荐：归一化后余弦相似度退化成
        点积，文本长短不再影响比较（为什么这样做见 LEARNING_NOTES 的归一化一节）。

        显存上限不在这里传：截断长度在模型加载时就设成实例级
        max_seq_length（见 _get_local_model），encode() 只负责算向量。这样
        显存上限只有一处代码在管，不会和调用参数漂移。

        文档侧和查询侧共用这一个方法，避免两处归一化设置漂移。
        """
        return self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )


class OpenAICompatibleEmbedding(BaseEmbedding):
    """任何 OpenAI 兼容的 /embeddings 接口。

    DeepSeek、智谱、通义、vLLM、Ollama 都兼容这个格式，所以只需要 base_url / api_key / model
    三个变量就能接上。故意不依赖 openai 库的强类型模型定义，用 duck typing，
    这样离线模式下不装 openai 也能跑。
    """

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        from openai import OpenAI  # 延迟导入，离线模式下不必装 openai 也能用

        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._dim: int | None = None

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def device_name(self) -> str:
        return "remote API"

    @property
    def dimensions(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed(["维度探测"])[0])   # 第一次调用时探测出来记住
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self._model, input=texts)
        ordered = sorted(response.data, key=lambda d: d.index)   # 接口不保证顺序，按 index 排
        vectors = [list(d.embedding) for d in ordered]
        self._dim = len(vectors[0])
        return vectors


def _resolve_device(requested: str) -> str:
    """auto → 有 CUDA 就用 CUDA，否则 CPU。

    批量算向量是纯计算密集的，GPU 比 CPU 快一个数量级。但没有 NVIDIA GPU
    或没装 CUDA 版 torch 时 CPU 也能跑，只是慢，所以这里降级而不是报错。
    """
    if requested != "auto":
        return requested
    import torch  # 延迟导入：local / openai 模式不需要 torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_dtype(device: str, requested: str) -> str:
    """auto → CUDA 用 fp16，否则 fp32。

    fp16 把 bge-m3 的权重从 2.17 GB 压到约 1.1 GB——实测加载后显存 reserved
    1090 MiB，fp32 会翻倍。BAAI 官方说明 bge-m3 支持半精度。

    已知代价：fp16 下 normalize_embeddings 之后向量的真实范数不是精确的 1。
    fp16 只有 10 位尾数，归一化标量做逐元素除法时每个商都被舍入。实测偏离
    1 在 2.6e-4 以内（scripts/smoke_real_embedding.py，覆盖 13 ~ 523 token）。
    这是确定的舍入、不是噪声——同一条文本重复编码的最大绝对元素差是 0.000e+00
    （scripts/diagnose_embedding_determinism.py）。

    CPU 没有 fp16 加速，而且 399.6 ms/条 对比 CUDA fp16 的 31.3 ms/条慢 12.8 倍，
    所以 CPU 保持 fp32。
    """
    if requested != "auto":
        return requested
    return "fp16" if device == "cuda" else "fp32"


def _get_local_model(model_name: str, device: str, dtype: str, max_length: int) -> Any:
    """带缓存地加载本地模型。同一个 (模型, 设备, 精度, 截断长度) 组合只加载一次。

    加载完立刻做两件事，都是在"模型刚生出来、还没被任何调用改过"的时候定下不变量：

    1. eval()  —— 关掉 dropout，保证同一条文本永远编出同一个向量。
       encode() 内部其实每次也会调 self.eval()（源码 model.py 里 encode() 带
       @torch.inference_mode() 装饰器），这里再设一次是为了让不变量出现在加载处，
       以后谁要在这份代码里改成训练用途，第一眼就会看到它。

    2. max_seq_length = max_length —— 显存上限。encode 会把整批 pad 到批内最长
       那一条，attention 矩阵按 batch × heads × seq² 增长：实测最长 chunk 有
       3654 token，batch=32 时这一批光 attention 就要 25 GiB，6 GB 的笔记本显卡
       直接 OOM 并挂死。封顶后只有 12 / 14303 条会被截尾。

       为什么用实例级属性而不是 encode(max_length=)：sentence-transformers 6.x
       的 encode() 会把不认识的额外关键字直接报 ValueError（本项目实测踩过），
       processing_kwargs 也能截断，但 max_seq_length 是 setter 直接写进 tokenizer
       的公共属性，行为等价且更简单。库默认 truncation="longest_first"，
       所以只要封顶生效就会真的截掉超长部分。

    max_length 也进缓存键：max_seq_length 是写进 tokenizer 的实例状态，两个不同
    上限的客户端若共用一个实例，后创建的那个会静默改掉前一个的截断上限，
    入库向量就会和记录在案的上限对不上。
    """
    key = (model_name, device, dtype, max_length)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    from sentence_transformers import SentenceTransformer  # 延迟导入，测试里是假模块

    kwargs: dict[str, Any] = {"device": device}
    if dtype == "fp16":
        # fp32 是模型的默认精度，不用显式传。这样 CPU 分支不需要 import torch，
        # 单元测试可以完全避开 torch 的几秒加载开销。
        import torch
        kwargs["model_kwargs"] = {"torch_dtype": torch.float16}

    model = SentenceTransformer(model_name, **kwargs)
    model.eval()
    model.max_seq_length = max_length
    _MODEL_CACHE[key] = model
    return model


def get_embedding_client(settings: Settings) -> BaseEmbedding:
    """工厂函数：根据 .env 里的 EMBEDDING_PROVIDER 决定用哪个实现。

    加新模型只需要在这里加分支，调用方（ingest / retriever / main）不用改。
    """
    if settings.embedding_provider == "openai":
        client: BaseEmbedding = OpenAICompatibleEmbedding(
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
            model=settings.embedding_model,
        )
    elif settings.embedding_provider == "local_semantic":
        client = LocalSemanticEmbedding(
            model_name=settings.local_embedding_model,
            device=settings.local_embedding_device,
            batch_size=settings.embedding_batch_size,
            query_prefix=settings.local_embedding_query_prefix,
            max_length=settings.local_embedding_max_length,
            dtype=settings.local_embedding_dtype,
        )
    else:
        client = HashingEmbedding(dimensions=settings.embedding_dim)

    print(
        f"Embedding provider: {settings.embedding_provider} / "
        f"Model: {client.model_name} / Device: {client.device_name}"
    )
    if isinstance(client, LocalSemanticEmbedding):
        # 这两个值会改变向量本身，所以在启动时就打印出来，
        # 而不是等改完 .env 才发现索引和模型对不上。
        print(
            f"  dtype: {client.dtype_name}  max_length: {client.max_length}  "
            f"batch_size: {settings.embedding_batch_size}"
        )
    return client
