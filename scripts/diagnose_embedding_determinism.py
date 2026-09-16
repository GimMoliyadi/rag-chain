"""诊断 bge-m3 在 fp16 下"同一条文本编码两次余弦只有 0.9878"的真正原因。

背景：
    fp32 下同文本重复编码的最大元素差是 2.01e-7；fp16 下余弦掉到 0.9878。
    余弦 0.9878 对应的单位向量距离是 0.156，差了六个数量级，不是"可以忽略的
    浮点噪声"。全量重建之前必须先定性，否则入库向量和查询向量之间有 1.2% 的
    不可复现偏差，检索等于在噪声里捞。

候选原因，逐个排除：
    1. dropout / 模型处于 train 模式   -> 看 model.training
    2. fp16 数值精度                   -> 同精度下多跑几次，方差应该稳定
    3. batch / padding 差异            -> 同一文本放进同一 batch vs 独立调用
    4. normalize 时机                  -> 归一化前对比，排除是归一化本身的问题
    5. 归一化后向量不是单位向量        -> 直接量 norm

用法：python scripts/diagnose_embedding_determinism.py
"""
from __future__ import annotations

import time

import torch
from sentence_transformers import SentenceTransformer

TEXT = "曲柄滑块机构在滑块到达极限位置时出现死点，此时机构无法靠自身驱动继续运动。" * 8
LONG_TEXT = TEXT * 4   # 约 1400 token，用于验证长文本上的差异是否更大
RUNS = 5


def say(msg: str = "") -> None:
    print(msg, flush=True)


def cosine(a, b) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


def max_abs_diff(a, b) -> float:
    return float(max(abs(x - y) for x, y in zip(a, b)))


def load(device: str, dtype: str) -> SentenceTransformer:
    """按精度加载同一个模型。fp32 不传 model_kwargs（它本来就是默认值）。"""
    kwargs = {"device": device}
    if dtype == "fp16":
        kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
    model = SentenceTransformer("BAAI/bge-m3", **kwargs)
    model.eval()
    return model


def report(title: str, model: SentenceTransformer, fp16: bool) -> None:
    say(f"\n{'=' * 74}")
    say(f"{title}")
    say(f"{'=' * 74}")
    say(f"device            = {model.device}")
    say(f"训练态 training   = {model.training}")
    say(f"各子模块 training = {sorted({m.training for m in model.modules()})}")
    params = [p for p in model.parameters()]
    say(f"参数 dtype        = {sorted({str(p.dtype) for p in params})}")
    say(f"max_seq_length    = {model.max_seq_length}")
    say(f"归一化            = normalize_embeddings=True（显式传给 encode）")

    base = {"normalize_embeddings": True, "convert_to_numpy": True,
            "show_progress_bar": False, "device": model.device}

    # A. 同一批里放两条完全相同的文本：同一次前向，排除跨调用因素
    batch = model.encode([TEXT, TEXT], batch_size=2, **base)
    say(f"\n[A] 同一 batch 内两条相同文本")
    say(f"    norm      = {cosine(batch[0], batch[0]):.10f} / {cosine(batch[1], batch[1]):.10f}")
    say(f"    cosine    = {cosine(batch[0], batch[1]):.10f}")
    say(f"    最大元素差 = {max_abs_diff(batch[0], batch[1]):.3e}")

    # B / C. 独立调用，重复 RUNS 次：量化跨调用波动
    say(f"\n[B/C] 独立调用 {RUNS} 次（batch_size=1）")
    vectors: list[list[float]] = []
    start = time.perf_counter()
    for i in range(RUNS):
        with torch.inference_mode():
            v = model.encode([TEXT], batch_size=1, **base)[0]
        vectors.append(list(v))
    elapsed = time.perf_counter() - start
    say(f"    耗时 {elapsed:.2f}s / {RUNS} 次 = {elapsed / RUNS * 1000:.1f} ms/次")
    say(f"    每条 norm        = {[f'{cosine(v, v):.8f}' for v in vectors]}")
    say(f"    各次 vs 第 1 次 cosine   = {[f'{cosine(v, vectors[0]):.8f}' for v in vectors]}")
    say(f"    各次 vs 第 1 次 最大元素差 = {[f'{max_abs_diff(v, vectors[0]):.3e}' for v in vectors]}")

    # D. 归一化前对比：区分是"前向本身有噪声"还是"归一化引入的"
    raw = [model.encode([TEXT], batch_size=1, normalize_embeddings=False,
                        convert_to_numpy=True, show_progress_bar=False)[0] for _ in range(2)]
    say(f"\n[D] 不归一化时")
    say(f"    cosine（未归一化）    = {cosine(raw[0], raw[1]):.10f}")
    say(f"    最大元素差（未归一化） = {max_abs_diff(raw[0], raw[1]):.3e}")
    say(f"    说明：未归一化余弦和归一化后余弦一致，就排除归一化时机的问题")

    # E. 长文本：如果差异随序列长度增大，指向 fp16 归约/累加
    long_vecs = [model.encode([LONG_TEXT], batch_size=1, **base)[0] for _ in range(2)]
    say(f"\n[E] 长文本（{len(LONG_TEXT)} 字符）重复编码")
    say(f"    cosine    = {cosine(long_vecs[0], long_vecs[1]):.10f}")
    say(f"    最大元素差 = {max_abs_diff(long_vecs[0], long_vecs[1]):.3e}")
    say(f"    （与短文本对比：差异随长度放大则指向 fp16 累加精度）")

    # F. 语义正确性：确定性问题不影响"能不能检索"这一层
    probe = {
        "曲柄滑块机构的死点": "机构死点",
        "POVM 与投影测量": "正算符值测度",
        "施工临时用电安全": "配电箱漏电保护",
    }
    say(f"\n[F] 语义正确性对照（余弦相似度）")
    for query, answer in probe.items():
        q = model.encode([query], batch_size=1, **base)[0]
        yes = model.encode([answer], batch_size=1, **base)[0]
        no = model.encode(["量子隧穿效应与势垒穿透"], batch_size=1, **base)[0]
        say(f"    {query} -> {answer:<10} = {cosine(q, yes):.4f}  |  无关文本 = {cosine(q, no):.4f}")


if __name__ == "__main__":
    say("诊断 bge-m3 的可复现性")
    say(f"torch = {torch.__version__}  cuda 可用 = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        say(f"GPU = {torch.cuda.get_device_name(0)}  显存 = "
            f"{torch.cuda.get_device_properties(0).total_memory / 2**20:.0f} MiB")

    report("① CUDA + fp16", load("cuda", "fp16"), True)
    report("② CUDA + fp32", load("cuda", "fp32"), False)
    report("③ CPU + fp32", load("cpu", "fp32"), False)
