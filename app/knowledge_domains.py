"""知识域映射：Vault 顶层目录 → (domain, source_type)。

映射关系只存在于 config/knowledge_domains.yaml，代码里不出现任何具体领域名：
换目录、加领域、改归类都只改 YAML。Python 这边只认一条规则——查表，查不到就 unknown。

匹配粒度是 file_path 的第一段（顶层目录名）。根目录散落的文件没有顶层目录，
天然落 unknown：它们大多是收集箱、模板这类杂项，本来就不属于任何知识域。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .router import QueryRouter

# 未映射路径的统一归宿。source_type 只是描述性元数据，不代表可信度（阶段十二的边界）。
UNKNOWN_DOMAIN = "unknown"
UNKNOWN_SOURCE_TYPE = "unknown"


@dataclass(frozen=True)
class DomainInfo:
    """一个顶层目录的知识域归属。"""

    domain: str
    source_type: str


def load_domain_mapping(config_path: Path) -> dict[str, DomainInfo]:
    """读 YAML 得到「顶层目录名 -> DomainInfo」。

    文件缺失或格式不对必须抛错而不是返回空表：空表会把整库静默刷成 unknown，
    那种"全部变成没归属"的事故没有任何报错，等发现时元数据已经写坏了。
    """
    config = _read_config(config_path)
    domains = config.get("knowledge_domains")
    if not isinstance(domains, dict):
        raise ValueError(f"{config_path} 缺少 knowledge_domains 段或格式不对")

    mapping: dict[str, DomainInfo] = {}
    for dir_name, spec in domains.items():
        if not isinstance(spec, dict) or "domain" not in spec:
            raise ValueError(f"{config_path} 里 {dir_name} 缺少 domain 字段")
        mapping[str(dir_name)] = DomainInfo(
            domain=str(spec["domain"]),
            source_type=str(spec.get("source_type", UNKNOWN_SOURCE_TYPE)),
        )
    return mapping


def load_router_terms(config_path: Path) -> dict[str, list[str]]:
    """读 YAML 的 router_terms 段：domain -> 触发术语列表。

    没有该段返回空表，Router 对任何问题都会给 global——这是配置缺失时的安全方向：
    宁可全库检索，不可按错误的路由过滤掉真答案。
    """
    config = _read_config(config_path)
    terms = config.get("router_terms") or {}
    if not isinstance(terms, dict):
        raise ValueError(f"{config_path} 的 router_terms 段格式不对，应为 domain -> 术语列表")
    return {str(domain): [str(term) for term in term_list] for domain, term_list in terms.items()}


def load_router(config_path: Path) -> tuple[QueryRouter, list[str]]:
    """一次读 YAML 组装「规则 Router + LLM fallback 的合法域清单」。

    合法域清单取 knowledge_domains 段里出现过的 domain 全集（去重、按名称排序），
    而不是 router_terms 的键——LLM fallback 应该能选到任何一个真实存在的域，
    包括那些没法靠术语识别的。CLI 和 API 各自组装时走这一个函数，
    两处的清单口径就不会分叉。
    """
    mapping = load_domain_mapping(config_path)
    allowed_domains = sorted({info.domain for info in mapping.values()})
    return QueryRouter(load_router_terms(config_path)), allowed_domains


def resolve_domain(file_path: str, mapping: dict[str, DomainInfo]) -> DomainInfo:
    """按 file_path 首段查映射，查不到返回 unknown。"""
    top_dir = file_path.split("/", 1)[0]
    return mapping.get(top_dir, DomainInfo(UNKNOWN_DOMAIN, UNKNOWN_SOURCE_TYPE))


def _read_config(config_path: Path) -> dict:
    if not config_path.is_file():
        raise FileNotFoundError(f"知识域配置文件不存在：{config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path} 的顶层结构应为键值映射")
    return raw
