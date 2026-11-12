"""检查生成回答的引用编号；不判断结论是否得到原文支持。"""
from __future__ import annotations

import re


_CODE = re.compile(
    r"```[\s\S]*?(?:```|\Z)|~~~[\s\S]*?(?:~~~|\Z)"
    r"|(?P<ticks>`+)[^\n]*?(?P=ticks)"
)
_CITATION = re.compile(r"(?<!\\)\[(\d+(?:\s*[,，]\s*\d+)*)\](?!\()")


def check_citations(answer: str, source_count: int) -> dict:
    text = _CODE.sub("", answer)
    cited = {
        int(number)
        for match in _CITATION.finditer(text)
        for number in re.split(r"\s*[,，]\s*", match.group(1))
    }
    available = set(range(1, source_count + 1))
    invalid = cited - available
    return {
        "status": "invalid" if invalid else "ok" if cited else "missing",
        "cited_indexes": sorted(cited),
        "invalid_indexes": sorted(invalid),
        "unused_indexes": sorted(available - cited),
    }


def citation_warning(check: dict | None) -> str | None:
    if check is None or check["status"] == "ok":
        return None
    if check["status"] == "invalid":
        indexes = "、".join(f"[{index}]" for index in check["invalid_indexes"])
        return f"警告：回答包含不存在的引用编号 {indexes}，请核对原文。"
    return "警告：回答未标注引用编号，无法将结论对应到具体来源。"
