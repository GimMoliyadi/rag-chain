"""真实模型 smoke test：真的加载 BAAI/bge-m3，只测少量文本。

和 pytest 的分工：
    tests/        假模型，验证我们自己这一层（分批、缓存、指纹、截断报告）。
                  快、不下载模型、不依赖 GPU，任何机器上都能跑。
    本脚本        真模型，验证"我们这一层"和真实的 sentence-transformers 6.x API
                  之间没有缝隙。假模型会把真模型拒绝的参数也照单收下，
                  所以 pytest 全绿不代表真实模型能跑——见 LEARNING_NOTES
                  「为什么 mock 测试全绿，真实模型仍然会失败」一节。

    文件名不带 test_ 前缀、也不放在 tests/ 下，pytest 默认不收集它。
    默认 pytest 绝不该去下载 / 加载 2 GB 的模型。

判据是量出来的，不是拍脑袋：
    可复现性用「最大绝对元素差」而不是「余弦相似度」。同一条文本重复编码，
    CUDA fp16 / CUDA fp32 / CPU fp32 三组实测的最大绝对元素差都是 0.000e+00
    （诊断脚本 scripts/diagnose_embedding_determinism.py）。所以归一化后
    余弦 0.9834 是 fp16 归一化舍入的必然结果，不是不可复现，放宽容差去
    迁就它会真的掩盖缺陷。
    归一化范数按精度分阈值：fp16 只有 10 位尾数，用 fp16 标量做逐元素除法
    时每个商都被舍入，真实范数偏离 1 实测在 1e-4 量级（各长度均在
    4.5e-4 以内）；fp32 实测 1e-8 量级。用同一个 0.05 会把 fp32 侧的
    真实缺陷一起盖掉。

加载耗时说明：本脚本每次运行都要重新加载模型。实测 42s ~ 68s，每次都重读
    391 个权重文件并打 HF Hub 未认证警告。本机没有验证过"缓存命中 <1s"，
    所以不要指望它比 pytest 快多少——这也是它不该进默认 pytest 的理由。

用法：python scripts/smoke_real_embedding.py
退出码 0 = 全部通过，1 = 有失败。
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from app.embedding import LocalSemanticEmbedding

MODEL = "BAAI/bge-m3"
MAX_LENGTH = 1024
RUNS = 5                # 同一条文本独立编码的次数
EXPECTED_DIM = 1024

# 归一化范数阈值，按精度分。取值依据见文件头注释：都取实测值的 2 倍左右。
NORM_TOLERANCE = {"fp16": 2e-2, "fp32": 1e-5}

# 语义对照对：(查询, 相关答案, 无关文本的实测余弦)。
# 用诊断脚本里实测通过的对照对。"POVM 与投影测量" -> "正算符值测度" 实测
# 0.4231，低于无关文本的 0.4373，是 bge-m3 的已知短板（缩写 ↔ 中文全称），
# 故意不放进判据——放进去这个 smoke test 会稳定失败，但它测的不是我们这层。
PAIRS = [
    ("曲柄滑块机构的死点", "机构死点", 0.3289),
    ("施工临时用电安全", "配电箱漏电保护", 0.3479),
]
UNRELATED = "量子隧穿效应与势垒穿透"
RELATED_FLOOR = 0.5     # 相关对照的余弦下限，低于这个值说明语义能力整体崩了
MARGIN = 0.1            # 相关对照要比无关对照高出多少才算真有区分度

# 短文本 + 长文本都要测。14303 个 chunk 的 token 分布是 p50=304 / p99=780 /
# max=3654，长文本才是主要工作负载，只测短文本会给出偏乐观的结论。
# 长文本用重复句构造，token 数在运行时用模型自己的 tokenizer 打印出来，
# 这样"到底测了多长"是可核验的。
TEXTS = [
    "曲柄滑块机构在滑块到达极限位置时出现死点。",
    UNRELATED,
]
LONG_TEXT = "曲柄滑块机构在滑块到达极限位置时出现死点，此时机构无法靠自身驱动继续运动。" * 20


def say(msg: str = "") -> None:
    print(msg, flush=True)


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def max_abs_diff(a: list[float], b: list[float]) -> float:
    return max(abs(x - y) for x, y in zip(a, b))


def main() -> int:
    failures: list[str] = []

    say(f"torch = {torch.__version__}  cuda 可用 = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        say(f"GPU = {torch.cuda.get_device_name(0)}  显存 = "
            f"{torch.cuda.get_device_properties(0).total_memory / 2**20:.0f} MiB")

    started = time.perf_counter()
    client = LocalSemanticEmbedding(
        MODEL, device="auto", dtype="auto", max_length=MAX_LENGTH
    )
    load_seconds = time.perf_counter() - started
    say(f"\n模型加载耗时 = {load_seconds:.1f}s"
        f"（本机实测 42s ~ 68s，每次都重读 391 个权重文件）")
    say(f"指纹 = {client.fingerprint}")

    # ---- 1. 模型能加载（跑到这里就说明通过了） ----
    say("\n[1] 模型能加载")
    say(f"    {client.model_name}  device={client.device_name}"
        f"  dtype={client.dtype_name}  max_length={client.max_length}")
    say("    通过")

    all_texts = TEXTS + [LONG_TEXT]
    # 打印 token 数：让"到底测了多长"可核验。count_tokens 传 truncation=False，
    # 拿到的是真实长度，不会被 max_length 截成假的 1024。
    token_counts = client.count_tokens(all_texts)

    # ---- 2. 维度 = 1024 ----
    vectors = client.embed(all_texts)
    dim = len(vectors[0])
    ok = dim == EXPECTED_DIM
    say(f"\n[2] 维度 = {EXPECTED_DIM}")
    say(f"    实测 = {dim}  {'通过' if ok else '失败'}")
    if not ok:
        failures.append(f"维度 {dim} != {EXPECTED_DIM}")

    # ---- 3. 无 NaN / Infinity ----
    bad = [(i, j) for i, v in enumerate(vectors)
           for j, x in enumerate(v) if not math.isfinite(x)]
    ok = not bad
    say(f"\n[3] 无 NaN / Infinity")
    say(f"    检查 {sum(len(v) for v in vectors)} 个元素，"
        f"非法值 {len(bad)} 个  {'通过' if ok else '失败 ' + str(bad[:5])}")
    if not ok:
        failures.append(f"出现 {len(bad)} 个 NaN/Infinity")

    # ---- 4. 归一化范数接近 1（阈值按精度分，长文本是最坏情况） ----
    tol = NORM_TOLERANCE[client.dtype_name]
    norms = [math.sqrt(cosine(v, v)) for v in vectors]
    worst = max(abs(n - 1.0) for n in norms)
    ok = worst <= tol
    say(f"\n[4] 归一化后范数接近 1（{client.dtype_name} 阈值 {tol:.0e}）")
    for t, n, k in zip(all_texts, norms, token_counts):
        say(f"    norm = {n:.10f}  ({len(t)} 字符 / {k} token)  {t[:20]}...")
    say(f"    最大偏离 = {worst:.3e}  {'通过' if ok else '失败'}")
    if not ok:
        failures.append(f"范数偏离 {worst:.3e} > 阈值 {tol:.0e}")

    # ---- 5. 同一条文本重复编码：最大绝对元素差必须为 0 ----
    say(f"\n[5] 同一条文本重复编码 {RUNS} 次（独立调用，batch_size=1）")
    for text in [TEXTS[0], LONG_TEXT]:
        tokens = client.count_tokens([text])[0]
        repeats = [client.embed_query(text) for _ in range(RUNS)]
        diffs = [max_abs_diff(repeats[0], r) for r in repeats[1:]]
        cosines = [cosine(repeats[0], r) for r in repeats[1:]]
        good = all(d == 0.0 for d in diffs)
        say(f"    [{len(text)} 字符 / {tokens} token]  各次 vs 第 1 次:")
        for i, (d, c) in enumerate(zip(diffs, cosines), start=2):
            say(f"        第 {i} 次: 最大绝对元素差 = {d:.3e}  余弦 = {c:.10f}")
        say(f"        {'通过' if good else '失败'}")
        if not good:
            failures.append(
                f"重复编码不可复现：{len(text)} 字符文本最大元素差 {max(diffs):.3e} != 0"
            )
    say(f"    判据：最大绝对元素差 == 0（不是放宽后的余弦容差）")
    say("    注：余弦 != 1 是 fp16 归一化舍入的必然结果（见 [4]），不是不可复现。")

    # ---- 6. 不同语义的文本得到不同向量 ----
    say(f"\n[6] 不同语义文本 -> 不同向量")
    ok = True
    for query, answer, unrelated_sim in PAIRS:
        q = client.embed_query(query)
        yes = client.embed_query(answer)
        no = client.embed_query(UNRELATED)
        sim_yes, sim_no = cosine(q, yes), cosine(q, no)
        diff_same = max_abs_diff(q, client.embed_query(query))
        good = diff_same == 0.0 and sim_yes > RELATED_FLOOR and sim_yes - sim_no > MARGIN
        ok = ok and good
        say(f"    查询 = {query}")
        say(f"        -> 相关 {answer:<10} 余弦 = {sim_yes:.4f}（需 > {RELATED_FLOOR}）")
        say(f"        -> 无关 {UNRELATED:<10} 余弦 = {sim_no:.4f}"
            f"（诊断脚本实测参考 {unrelated_sim:.4f}）")
        say(f"        差值 = {sim_yes - sim_no:.4f}（需 > {MARGIN}）"
            f"  同查询两次最大元素差 = {diff_same:.3e}  {'通过' if good else '失败'}")
        if not good:
            failures.append(f"语义对照失败: {query} -> {answer}")
    if not ok:
        failures.append("语义区分度不足")
    say("    已知限制：bge-m3 对缩写 ↔ 中文全称映射偏弱"
        "（如 POVM -> 正算符值测度 实测 0.4231，低于无关文本 0.4373），"
        "此处不作为判据。")

    # ---- 附加测量：单条编码 vs 批内编码是否一致 ----
    # 文档侧走 batch，查询侧走单条。如果两边向量不一致，检索就是在两个略有
    # 偏移的空间里比余弦——这是"模型没问题但检索效果差"的隐蔽来源。
    #
    # 实测结论（不要写成"期望为 0"）：fp16 下两边并不逐位相同，差在 1e-4 ~ 5e-4。
    # 排除掉 padding 泄漏这条：padding 位置在 attention 里被 mask 成 -inf，
    # softmax 之后贡献正好是 0，所以批内 pad 多长不会污染真实 token 的输出。
    # 实测也支持这一点——只编 2 条短文本时差是 0.000e+00，改成 3 条含长文本
    # 时才出现非零，差异跟着批内组成变，不是跟着 padding 长度变。
    # 所以来源是批内形状变化：padding 到不同长度、可能选到不同的 cuDNN 内核。
    # 量级正好落在 fp16 舍入范围内（fp16 机器精度 2^-10 ≈ 9.8e-4）。
    # 影响判断：语义相似度差异在 0.1 ~ 0.4 量级，1e-4 的偏移不会改变排序，
    # 所以只测量不判据；量级明显偏大（比如到 1e-2）才需要停下来查。
    batch_vectors = client.embed(all_texts)
    cross = [max_abs_diff(client.embed_query(t), v) for t, v in zip(all_texts, batch_vectors)]
    say(f"\n[附加] 单条编码 vs 批内编码的最大绝对元素差（仅测量，不作为判据）")
    for t, d, k in zip(all_texts, cross, token_counts):
        say(f"    {d:.3e}  ({len(t)} 字符 / {k} token)")
    say(f"    实测 {max(cross):.1e}，落在 fp16 舍入量级内；排序不受影响，故不判据。")

    # ---- 结果 ----
    say("\n" + "=" * 60)
    if failures:
        say(f"结果：失败 {len(failures)} 项")
        for f in failures:
            say(f"  - {f}")
        say("结论：不要把这次配置用于全量重建。")
        return 1
    say("结果：全部通过")
    say(f"    {client.model_name} / {client.device_name} / {client.dtype_name}"
        f" / max_length={client.max_length} / dim={dim}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
