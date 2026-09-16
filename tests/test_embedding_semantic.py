"""测试：LocalSemanticEmbedding 本地语义 embedding provider。

这些测试不能依赖下载多 GB 的模型，所以把 sentence_transformers 换成一个
假的模块（塞进 sys.modules），只验证我们自己这一层的逻辑：
模型缓存、分批、维度探测、查询前缀、设备选择、指纹校验。

覆盖清单：
    1. 工厂能按 .env 建出 local_semantic provider，配置校验拦住非法值
    2. 同一个 (模型, 设备, 精度) 组合只加载一次
    3. embed() 按 EMBEDDING_BATCH_SIZE 分批喂给模型
    4. embed_query 返回单条向量
    5. 维度与模型实际输出一致，且确实做了归一化
    6. 换了模型 / 精度 / 截断长度能被发现（EmbeddingMismatchError），不会静默继续
    7. 显存上限真的传进 encode，fp16 真的传给 SentenceTransformer
    8. 哈希 fallback 不受影响
    9. OpenAI 兼容 provider 不受影响
    10. 重建向量索引只动 Chroma，SQLite 一个字都不改
    11. 超长 chunk 被点名报告，不会静默截断
"""
from __future__ import annotations

import hashlib
import sys
import types

import numpy as np
import pytest

import app.embedding as emb
from app.config import Settings
from app.retriever import EmbeddingMismatchError, Retriever
from app.vectorstore import ChromaStore

# 假模型的输出维度。真模型是 1024，这里用 8 就够了——测的是我们的代码，不是模型
FAKE_DIM = 8


class FakeModel:
    """假的 SentenceTransformer 实例。

    关键行为要和真模型一致：
        - 同一个文本永远编出同一个向量（确定性）
        - normalize_embeddings=True 时输出单位向量
        - 单条字符串输入返回 (dim,)，列表输入返回 (n, dim)
    另外记录每次收到的文本数，用来验证分批。
    """

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []
        self.normalize_flags: list[bool] = []
        self.encoded: list[str] = []
        # 默认值对齐真模型（bge-m3 是 8192），这样"有没有被我们改掉"看得出差别
        self.max_seq_length = 8192
        self.training = False
        # count_tokens 走这里。真模型是 XLM-RoBERTa tokenizer，假模型按字符数算
        self.tokenizer = _fake_tokenize

    def eval(self):
        """真模型：关掉 dropout。假模型只需要把 training 标志置 False。"""
        self.training = False
        return self

    def encode(self, texts, batch_size=32, normalize_embeddings=False,
               show_progress_bar=True, convert_to_numpy=True):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        self.batch_sizes.append(len(items))
        self.normalize_flags.append(bool(normalize_embeddings))
        self.encoded.extend(items)

        vectors = []
        for text in items:
            vec = _fake_vector(text, FAKE_DIM)
            if normalize_embeddings:
                vec = vec / np.linalg.norm(vec)
            vectors.append(vec)

        out = np.stack(vectors)
        return out[0] if single else out


def _fake_tokenize(texts: list[str], truncation: bool = True) -> dict:
    """假 tokenizer：token 数 = 字符数，返回真 tokenizer 的形状 {"input_ids": [...]}。

    只为让 count_tokens 有东西可数——测试关心的是"数没数对、有没有报超长的"，
    不是分词算法本身。字符数和字符串长度成正比，超长文本容易构造。

    这里不真的按 truncation 截断：count_tokens 传的是 truncation=False，
    测的是"数没数对"，不是截断逻辑。真 tokenizer 的截断由 encode() 自己做。
    """
    return {"input_ids": [[0] * len(t) for t in texts]}


def _fake_vector(text: str, dim: int) -> np.ndarray:
    """由文本内容决定向量：同一个文本映射到同一个向量。"""
    digest = hashlib.md5(text.encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:4], "big") % (2**31))
    return rng.standard_normal(dim).astype(np.float32)


@pytest.fixture
def fake_st(monkeypatch):
    """注入假的 sentence_transformers 模块，并清空模型缓存。

    用 sys.modules 注入的原因：`from sentence_transformers import SentenceTransformer`
    会先查 sys.modules，所以走的是真实的 import 语句，只是模块本身是假的。
    这样测试既快，又不需要真的下载模型。
    """
    instances: list[FakeModel] = []

    def fake_sentence_transformer(model_name, device="cpu", model_kwargs=None):
        model = FakeModel()
        model.model_name = model_name
        model.device = device
        model.model_kwargs = dict(model_kwargs or {})
        instances.append(model)
        return model

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=fake_sentence_transformer),
    )
    emb._MODEL_CACHE.clear()
    yield instances
    emb._MODEL_CACHE.clear()


@pytest.fixture
def semantic_settings(tmp_path):
    """一份 local_semantic 模式的配置，batch_size 故意设成 7 好验证分批。"""
    return Settings(
        vault_path=str(tmp_path / "vault"),
        db_path=str(tmp_path / "test.db"),
        chroma_path=str(tmp_path / "chroma"),
        embedding_provider="local_semantic",
        local_embedding_model="BAAI/bge-m3",
        local_embedding_device="cpu",
        local_embedding_query_prefix="",
        embedding_batch_size=7,
        llm_provider="mock",
    )


# ---- 1. 工厂能建出正确的 provider ----


def test_factory_creates_local_semantic(semantic_settings, fake_st):
    client = emb.get_embedding_client(semantic_settings)
    assert isinstance(client, emb.LocalSemanticEmbedding)
    assert client.model_name == "BAAI/bge-m3"
    assert client.device_name == "cpu"
    # .env 里的模型名和设备真的传进去了
    assert (fake_st[0].model_name, fake_st[0].device) == ("BAAI/bge-m3", "cpu")


def test_factory_ignores_model_name_when_not_local_semantic(semantic_settings, fake_st):
    semantic_settings.embedding_provider = "local"
    client = emb.get_embedding_client(semantic_settings)
    assert isinstance(client, emb.HashingEmbedding)
    assert fake_st == []      # local 模式绝不能碰本地模型


def test_validate_rejects_bad_embedding_provider():
    with pytest.raises(ValueError, match="EMBEDDING_PROVIDER"):
        Settings(embedding_provider="bogus").validate_providers()


def test_validate_rejects_bad_local_device():
    with pytest.raises(ValueError, match="LOCAL_EMBEDDING_DEVICE"):
        Settings(local_embedding_device="tpu").validate_providers()


def test_validate_rejects_bad_batch_size():
    with pytest.raises(ValueError, match="EMBEDDING_BATCH_SIZE"):
        Settings(embedding_batch_size=0).validate_providers()


def test_validate_rejects_bad_local_dtype():
    with pytest.raises(ValueError, match="LOCAL_EMBEDDING_DTYPE"):
        Settings(local_embedding_dtype="fp64").validate_providers()


def test_validate_rejects_bad_max_length():
    with pytest.raises(ValueError, match="LOCAL_EMBEDDING_MAX_LENGTH"):
        Settings(local_embedding_max_length=0).validate_providers()


# ---- 2. 模型只加载一次 ----


def test_model_loaded_only_once(fake_st):
    a = emb.LocalSemanticEmbedding("m1", device="cpu")
    b = emb.LocalSemanticEmbedding("m1", device="cpu")
    assert a._model is b._model
    assert len(fake_st) == 1


def test_different_model_or_device_loads_separately(fake_st):
    emb.LocalSemanticEmbedding("m1", device="cpu")
    emb.LocalSemanticEmbedding("m1", device="cuda")
    emb.LocalSemanticEmbedding("m2", device="cpu")
    assert len(fake_st) == 3


# ---- 3. 分批 ----


def test_embed_batches_by_batch_size(fake_st):
    """分批发生在 provider 内部，调用方整批发来就行。"""
    client = emb.LocalSemanticEmbedding("m1", device="cpu", batch_size=7)
    texts = [f"第 {i} 条笔记" for i in range(100)]
    vectors = client.embed(texts)

    assert len(vectors) == len(texts)
    assert fake_st[0].batch_sizes == [7] * 14 + [2]   # 100 = 14*7 + 2
    assert all(len(v) == FAKE_DIM for v in vectors)


def test_embed_is_deterministic(fake_st):
    client = emb.LocalSemanticEmbedding("m1", device="cpu", batch_size=3)
    texts = ["甲", "乙", "丙"]
    assert client.embed(texts) == client.embed(texts)


def test_embed_empty_list_does_not_call_model(fake_st):
    client = emb.LocalSemanticEmbedding("m1", device="cpu")
    assert client.embed([]) == []
    assert fake_st[0].batch_sizes == []


# ---- 4 / 5. embed_query 与维度 ----


def test_embed_query_returns_single_vector(fake_st):
    client = emb.LocalSemanticEmbedding("m1", device="cpu")
    vec = client.embed_query("曲柄滑块机构什么时候出现死点")
    assert isinstance(vec, list)
    assert len(vec) == FAKE_DIM
    assert all(isinstance(x, float) for x in vec)


def test_dimensions_match_model_output(fake_st):
    client = emb.LocalSemanticEmbedding("m1", device="cpu")
    assert client.dimensions == FAKE_DIM


def test_vectors_are_normalized(fake_st):
    """BGE 官方推荐 normalize_embeddings=True，必须真的传进去了。"""
    client = emb.LocalSemanticEmbedding("m1", device="cpu")
    assert client.embed_query("任意文本")
    assert all(fake_st[0].normalize_flags)
    vec = client.embed_query("任意文本")
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-6


def test_query_prefix_only_applies_to_query_side(fake_st):
    """前缀只加在查询上，文档侧绝不能被污染。"""
    client = emb.LocalSemanticEmbedding("m1", device="cpu", query_prefix="检索指令：")
    client.embed_query("死点")
    client.embed(["文档正文"])
    assert fake_st[0].encoded == ["检索指令：死点", "文档正文"]


def test_query_prefix_empty_means_same_as_embed(fake_st):
    client = emb.LocalSemanticEmbedding("m1", device="cpu")
    assert client.embed_query("死点") == client.embed(["死点"])[0]


# ---- 设备选择 ----


def test_resolve_device_auto_uses_cuda_when_available(monkeypatch):
    _fake_torch(monkeypatch, available=True)
    assert emb._resolve_device("auto") == "cuda"


def test_resolve_device_auto_falls_back_to_cpu(monkeypatch):
    _fake_torch(monkeypatch, available=False)
    assert emb._resolve_device("auto") == "cpu"


def test_resolve_device_explicit_value_is_not_ignored(monkeypatch):
    """显式指定 cpu 时，就算有 GPU 也不能用 GPU——用户说 CPU 就是要 CPU。"""
    _fake_torch(monkeypatch, available=True)
    assert emb._resolve_device("cpu") == "cpu"
    assert emb._resolve_device("mps") == "mps"


def test_resolve_dtype_auto_uses_half_precision_on_cuda(monkeypatch):
    """CUDA 上自动 fp16：bge-m3 权重从 2.17 GB 压到约 1.1 GB，6 GB 显卡才装得下。"""
    _fake_torch(monkeypatch, available=True)
    assert emb._resolve_dtype("cuda", "auto") == "fp16"


def test_resolve_dtype_auto_stays_float32_on_cpu(monkeypatch):
    """CPU 没有 fp16 加速，归一化舍入误差也更明显，所以保持 fp32。"""
    _fake_torch(monkeypatch, available=False)
    assert emb._resolve_dtype("cpu", "auto") == "fp32"


def test_resolve_dtype_explicit_value_is_not_ignored():
    assert emb._resolve_dtype("cpu", "fp16") == "fp16"
    assert emb._resolve_dtype("cuda", "fp32") == "fp32"


def test_half_precision_is_actually_applied(fake_st, monkeypatch):
    """fp16 不能只写在配置里，必须真的作为 model_kwargs 传进 SentenceTransformer。"""
    _fake_torch(monkeypatch, available=True)
    emb.LocalSemanticEmbedding("m1", device="cuda")
    assert fake_st[0].model_kwargs == {"torch_dtype": "fake-fp16"}


def test_float32_does_not_pass_model_kwargs(fake_st):
    """fp32 是默认精度，不传 model_kwargs，这样 CPU 分支连 torch 都不用导入。"""
    emb.LocalSemanticEmbedding("m1", device="cpu")
    assert fake_st[0].model_kwargs == {}


def test_max_length_is_set_on_the_model(fake_st):
    """显存上限必须真的写进模型，不能只留在配置里。

    不写的话最长那条 chunk 还是会把整批 pad 到 3654 token，batch=32 下
    光 attention 就要 25 GB，6 GB 的显卡直接 OOM。

    断言的是"模型上的值被改掉了"而不是"我们传了什么参数"：默认值是 8192，
    改成 768 才算真的生效。只存了个变量、没写进模型，等于什么都没做。
    """
    emb.LocalSemanticEmbedding("m1", device="cpu", max_length=768)
    assert fake_st[0].max_seq_length == 768


def test_model_loads_separately_per_max_length(fake_st):
    """不同截断长度要用不同实例：max_seq_length 是写进 tokenizer 的实例状态，
    共用一个实例会让后创建的客户端静默改掉前一个的上限。"""
    emb.LocalSemanticEmbedding("m1", device="cpu", max_length=512)
    emb.LocalSemanticEmbedding("m1", device="cpu", max_length=1024)
    assert [m.max_seq_length for m in fake_st] == [512, 1024]


def test_model_is_loaded_in_eval_mode(fake_st):
    """加载完就得是 eval：train 模式下 dropout 会让同一条文本编出不同向量。"""
    emb.LocalSemanticEmbedding("m1", device="cpu").embed(["任意文本"])
    assert fake_st[0].training is False


def _fake_torch(monkeypatch, available: bool) -> None:
    """注入假 torch。float16 用字符串代替，好让断言不依赖真 torch。"""
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: available),
            float16="fake-fp16",
        ),
    )


# ---- 截断报告：超长 chunk 不能静默丢内容 ----


def test_count_tokens_returns_one_length_per_text(fake_st):
    """每条文本一个 token 数，顺序要和输入对齐——错一条后面全错位。"""
    client = emb.LocalSemanticEmbedding("m1", device="cpu")
    assert client.count_tokens(["甲", "乙丙丁", ""]) == [1, 3, 0]


def test_report_truncated_lists_every_long_chunk(fake_st, capsys):
    """超长 chunk 必须点名（chunk_id / 文档 / 实际 token 数），不能静默截断。

    只报"3 条被截断了"不够——调用方需要知道是哪几条、内容缺了多少。
    """
    from app.ingest import report_truncated

    client = emb.LocalSemanticEmbedding("m1", device="cpu", max_length=768)
    rows = [
        {"chunk_id": 1, "title": "短文档", "content": "甲" * 100},
        {"chunk_id": 2, "title": "长文档一", "content": "乙" * 800},
        {"chunk_id": 3, "title": "长文档二", "content": "丙" * 5000},
    ]
    assert report_truncated(client, rows) == 2

    out = capsys.readouterr().out
    assert "chunk_id=2" in out and "长文档一" in out and "token_length=800" in out
    assert "chunk_id=3" in out and "token_length=5000" in out
    assert "chunk_id=1" not in out      # 没超长的不能出现在 warning 里


def test_hashing_has_no_token_length_concept():
    """哈希向量没有序列长度概念：不该报错，也不该误报截断。"""
    from app.ingest import report_truncated

    hashing = emb.HashingEmbedding()
    assert hashing.max_length is None
    assert hashing.count_tokens(["任意文本"]) is None
    assert report_truncated(hashing, [{"content": "任意长度文本" * 100}]) == 0


# ---- 6. 换模型必须被发现 ----


def _seeded_store(tmp_path, embedding):
    """造一个已入库的向量库，返回 store。"""
    store = ChromaStore(tmp_path / "chroma")
    store.upsert_chunks([_chunk_row(embedding.embed_query("死点"))])
    store.write_index_info(embedding.fingerprint)
    return store


def test_model_change_is_detected(fake_st, tmp_path):
    """昨天 bge-small、今天 bge-m3，但 Chroma 没重建——必须报错，不能静默继续。"""
    old = emb.LocalSemanticEmbedding("bge-small-zh-v1.5", device="cpu")
    store = _seeded_store(tmp_path, old)

    new = emb.LocalSemanticEmbedding("BAAI/bge-m3", device="cpu")
    retriever = Retriever(embedding=new, store=store)
    with pytest.raises(EmbeddingMismatchError) as err:
        retriever.search("死点")
    assert "rebuild" in str(err.value)


def test_missing_fingerprint_is_detected(fake_st, tmp_path):
    """旧版代码建的库没有指纹文件，同样要报错。"""
    store = ChromaStore(tmp_path / "chroma")
    store.upsert_chunks([_chunk_row([0.1] * FAKE_DIM)])

    retriever = Retriever(embedding=emb.LocalSemanticEmbedding("BAAI/bge-m3", device="cpu"), store=store)
    with pytest.raises(EmbeddingMismatchError) as err:
        retriever.search("死点")
    assert "rebuild" in str(err.value)


def test_matching_fingerprint_passes(fake_st, tmp_path):
    embedding = emb.LocalSemanticEmbedding("BAAI/bge-m3", device="cpu")
    store = _seeded_store(tmp_path, embedding)
    hits = Retriever(embedding=embedding, store=store).search("死点", top_k=1)
    assert len(hits) == 1
    assert 0.0 <= hits[0].score <= 1.0


def test_fingerprint_records_everything_that_changes_vectors(fake_st):
    """指纹要覆盖所有会改变向量的东西，否则改 .env 后旧索引会被悄悄继续查询。"""
    client = emb.LocalSemanticEmbedding("BAAI/bge-m3", device="cpu", max_length=512)
    assert client.fingerprint == {
        "model": "BAAI/bge-m3",
        "dimensions": FAKE_DIM,
        "dtype": "fp32",
        "max_length": 512,
    }


def test_dtype_change_is_detected(fake_st, monkeypatch, tmp_path):
    """fp16 和 fp32 算出的浮点数不同：改精度等于换向量，必须重建。

    这里模型名和维度都相同，唯一的差异就是 dtype——验证新增字段真的生效。
    """
    _fake_torch(monkeypatch, available=True)
    old = emb.LocalSemanticEmbedding("BAAI/bge-m3", device="cpu", dtype="fp32")
    store = _seeded_store(tmp_path, old)

    new = emb.LocalSemanticEmbedding("BAAI/bge-m3", device="cuda")
    with pytest.raises(EmbeddingMismatchError) as err:
        Retriever(embedding=new, store=store).search("死点")
    assert "rebuild" in str(err.value)


def test_max_length_change_is_detected(fake_st, tmp_path):
    """改了截断长度也算换向量：超长 chunk 的尾部内容变了，向量就变了。"""
    old = emb.LocalSemanticEmbedding("BAAI/bge-m3", device="cpu", max_length=512)
    store = _seeded_store(tmp_path, old)

    new = emb.LocalSemanticEmbedding("BAAI/bge-m3", device="cpu", max_length=1024)
    with pytest.raises(EmbeddingMismatchError) as err:
        Retriever(embedding=new, store=store).search("死点")
    assert "rebuild" in str(err.value)


# ---- 7. 哈希 fallback 不受影响 ----


def test_hashing_fallback_still_works():
    h = emb.HashingEmbedding(dimensions=256)
    assert h.model_name == "hashing-256d"
    assert h.device_name == "CPU"
    assert h.dimensions == 256

    vec = h.embed_query("POVM 正算符值测度")
    assert len(vec) == 256
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-6

    assert h.embed(["a"]) == h.embed(["a"])                    # 确定性
    assert h.embed_query("a") == h.embed(["a"])[0]             # 走基类默认实现
    assert h.embed([]) == []


# ---- 8. OpenAI 兼容 provider 不受影响 ----


def test_openai_provider_still_works(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", _fake_openai(dim=512))

    client = emb.OpenAICompatibleEmbedding(
        base_url="http://example.invalid", api_key="test", model="text-embedding-3-small"
    )
    assert client.model_name == "text-embedding-3-small"
    assert client.device_name == "remote API"
    assert client.dimensions == 512

    vectors = client.embed(["甲", "乙"])
    assert len(vectors) == 2
    # 假接口故意乱序返回，这里确认按 index 排回原顺序
    assert vectors[0][0] == pytest.approx(0.1)
    assert vectors[1][0] == pytest.approx(0.2)


def test_openai_provider_requires_credentials():
    with pytest.raises(ValueError, match="EMBEDDING_BASE_URL"):
        Settings(embedding_provider="openai", embedding_base_url="", embedding_api_key="").validate_providers()


def _fake_openai(dim: int):
    """假的 openai 模块：故意按乱序返回 embedding，用来验证按 index 排序的逻辑。"""
    def create(**kwargs):
        inputs = list(kwargs["input"])
        data = [
            types.SimpleNamespace(index=i, embedding=[0.1 * (i + 1)] * dim)
            for i in range(len(inputs))
        ]
        data.reverse()
        return types.SimpleNamespace(data=data)

    client_cls = lambda base_url, api_key: types.SimpleNamespace(
        embeddings=types.SimpleNamespace(create=create)
    )
    return types.SimpleNamespace(OpenAI=client_cls)


# ---- 重建向量索引：只能动 Chroma，不能碰 SQLite ----


def test_rebuild_vectors_only_touches_chroma(fake_vault, tmp_path, fake_st):
    """换模型后重建：SQLite 里的 chunk 一个字都不能变，变的只有 Chroma。"""
    from app.ingest import rebuild_vectors, run_ingest

    settings = Settings(
        vault_path=str(fake_vault),
        db_path=str(tmp_path / "test.db"),
        chroma_path=str(tmp_path / "chroma"),
        embedding_provider="local",
        llm_provider="mock",
    )
    store = ChromaStore(settings.chroma_path_resolved)
    try:
        run_ingest(settings, emb.HashingEmbedding(dimensions=256), store)
        assert store.read_index_info() == {
            "model": "hashing-256d", "dimensions": 256
        }
    finally:
        # 关掉 handle：重建要 rename 这个目录，Windows 上文件被占用时 rename 会失败
        store.close()

    before = _sqlite_snapshot(settings)

    semantic = emb.LocalSemanticEmbedding("BAAI/bge-m3", device="cpu")
    count = rebuild_vectors(settings, semantic)

    store = ChromaStore(settings.chroma_path_resolved)
    try:
        assert count == before["chunk_count"]
        assert store.count() == before["chunk_count"]
        assert store.read_index_info() == semantic.fingerprint
        assert _sqlite_snapshot(settings) == before      # SQLite 完全没动
    finally:
        store.close()


def test_rebuild_vectors_failure_leaves_production_index_untouched(fake_vault, tmp_path, fake_st, monkeypatch):
    """构建中途校验失败：正式索引原样不动，临时目录留下供诊断。"""
    from app import ingest as ingest_mod
    from app.ingest import rebuild_vectors, run_ingest

    settings = Settings(
        vault_path=str(fake_vault),
        db_path=str(tmp_path / "test.db"),
        chroma_path=str(tmp_path / "chroma"),
        embedding_provider="local",
        llm_provider="mock",
    )
    store = ChromaStore(settings.chroma_path_resolved)
    try:
        run_ingest(settings, emb.HashingEmbedding(dimensions=256), store)
    finally:
        store.close()

    old_fingerprint = {
        "model": "hashing-256d", "dimensions": 256
    }
    before = _sqlite_snapshot(settings)

    # 故意让校验失败，模拟 count 不符 / 索引不可应答
    monkeypatch.setattr(ingest_mod, "_verify_build", lambda *a, **k: False)
    assert rebuild_vectors(settings, emb.LocalSemanticEmbedding("BAAI/bge-m3", device="cpu")) == 0

    build_dir = settings.chroma_path_resolved.parent / (
        settings.chroma_path_resolved.name + "_building"
    )
    assert build_dir.is_dir()                       # 临时目录保留供诊断

    store = ChromaStore(settings.chroma_path_resolved)
    try:
        assert store.read_index_info() == old_fingerprint   # 指纹没被半成品覆盖
        assert store.count() == before["chunk_count"]       # 旧索引还是旧的，没被替换
    finally:
        store.close()
    assert _sqlite_snapshot(settings) == before


def test_rebuild_vectors_noop_on_empty_db(tmp_path, fake_st):
    from app.ingest import rebuild_vectors

    settings = Settings(
        vault_path=str(tmp_path),
        db_path=str(tmp_path / "test.db"),
        chroma_path=str(tmp_path / "chroma"),
        embedding_provider="local",
    )
    assert rebuild_vectors(settings, emb.LocalSemanticEmbedding("m1", device="cpu")) == 0


def test_store_clear_removes_vectors_but_keeps_file(tmp_path):
    store = ChromaStore(tmp_path / "chroma")
    store.upsert_chunks([_chunk_row([0.1] * FAKE_DIM)])
    store.upsert_chunks([_chunk_row([0.2] * FAKE_DIM, chunk_id=2)])
    assert store.count() == 2
    store.clear()
    assert store.count() == 0
    # clear 只删 collection，不动指纹文件——指纹要在全部写完之后再写
    store.write_index_info({"model": "m1", "dimensions": FAKE_DIM})
    assert store.read_index_info() == {"model": "m1", "dimensions": FAKE_DIM}


# ---- 工具 ----


def _chunk_row(vector: list[float], chunk_id: int = 1) -> dict:
    """构造一条能通过 ChromaStore.upsert_chunks 的行。"""
    return {
        "chunk_id": chunk_id,
        "document_id": 1,
        "chunk_index": 0,
        "content": "曲柄滑块机构在滑块到达极限位置时出现死点。",
        "title": "曲柄滑块机构",
        "file_path": "机械/曲柄滑块机构.md",
        "heading": "死点位置",
        "source": "机械原理",
        "folder": "机械",
        "tags": ["机械", "机构学"],
        "embedding": vector,
    }


def _sqlite_snapshot(settings: Settings):
    """把 SQLite 里的文档和 chunk 全文抓出来做快照，用来证明重建没有改它。"""
    from app.database import connect, init_db

    conn = connect(settings.db_path_resolved)
    init_db(conn)
    try:
        return {
            "documents": [tuple(r) for r in conn.execute(
                "SELECT id, file_path, content_hash FROM documents ORDER BY id")],
            "chunks": [tuple(r) for r in conn.execute(
                "SELECT id, document_id, chunk_index, content FROM chunks ORDER BY id")],
            "chunk_count": conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
        }
    finally:
        conn.close()
