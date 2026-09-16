"""LLM 客户端，设计成可替换的。

和 embedding 一样收口成一个接口，换模型只改 .env。

三个实现：

    BaseLLM
        ├── OpenAICompatibleLLM      openai      任意 OpenAI 兼容 /chat/completions
        ├── AnthropicCompatibleLLM   anthropic   Anthropic Messages API（官方或兼容代理）
        └── MockLLM                  mock        离线回显，不发任何请求

对外只有 chat(messages) 和 model_name。messages 统一用 OpenAI 风格的
[{"role": "system", ...}, {"role": "user", ...}]，各实现在自己的边界上做格式适配，
rag.py 不需要知道底下接的是哪种 API。
"""
from __future__ import annotations

import os

from .config import Settings


class BaseLLM:
    """所有 LLM 实现的共同接口：收一组 messages，回一段文本。"""

    def chat(self, messages: list[dict]) -> str:
        raise NotImplementedError

    @property
    def model_name(self) -> str:
        raise NotImplementedError


class OpenAICompatibleLLM(BaseLLM):
    """任意 OpenAI 兼容 /chat/completions 接口。"""

    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 0.2):
        from openai import OpenAI  # 延迟导入，mock 模式下不需要 openai

        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._temperature = temperature

    @property
    def model_name(self) -> str:
        return self._model

    def chat(self, messages: list[dict]) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
        )
        return response.choices[0].message.content or ""


class AnthropicCompatibleLLM(BaseLLM):
    """Anthropic Messages API 端点：官方 API 或 CC Switch 这类兼容代理。

    和 OpenAICompatibleLLM 的差别不在模型，而在 wire 格式。Anthropic 不允许
    messages 数组里出现 role="system"——system 必须是请求的顶层参数，而且首条
    message 必须是 user。本项目 rag.py 拼出来的是 OpenAI 风格的 [system, user]，
    所以 chat() 先在边界上把 system 条目摘出去再交给 SDK。适配只发生在这一个
    类里，rag.py 完全不用改。

    凭证刻意不走 .env：SDK 的零参构造自己从 ANTHROPIC_BASE_URL /
    ANTHROPIC_AUTH_TOKEN 解析，token 因此既不进代码也不落盘。这也是能直接复用
    Claude Code 进程环境的前提。模型 ID 的优先级是构造参数（.env 的 LLM_MODEL）
    高于 ANTHROPIC_MODEL 环境变量。

    不传 temperature / thinking 这类参数：本端点是第三方兼容代理，只验证过最简
    请求（model + max_tokens + messages），加 Anthropic 专有参数很可能直接 400。
    少传参数就是少一个失败点。
    """

    def __init__(self, model: str | None = None, max_tokens: int = 4096) -> None:
        import anthropic  # 延迟导入，mock / openai 模式下不需要 anthropic

        self._client = anthropic.Anthropic()
        self._model = (model or os.environ.get("ANTHROPIC_MODEL", "")).strip()
        self._max_tokens = max_tokens

    @property
    def model_name(self) -> str:
        return self._model

    def chat(self, messages: list[dict]) -> str:
        # 拆 system：Anthropic 要求它作顶层参数，messages 里只留 user / assistant。
        system_parts: list[str] = []
        api_messages: list[dict] = []
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                api_messages.append({"role": msg["role"], "content": msg["content"]})

        create_kwargs: dict[str, object] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": api_messages,
        }
        if system_parts:
            create_kwargs["system"] = "\n\n".join(system_parts)

        response = self._client.messages.create(**create_kwargs)
        # 响应的 content 是 block 列表，text / thinking / tool_use 混排，
        # 必须按 type 挑，不能假设第一个块就是文本。
        return "".join(block.text for block in response.content if block.type == "text")


class MockLLM(BaseLLM):
    """离线回显实现。

    不调用任何外部服务，直接把喂给它的 context 原样吐回来（加个说明）。
    作用只有一个：没有 API Key 时也能验证 /api/chat 的接口结构、
    Prompt 拼装、引用编号、返回格式是不是都正确。
    看到回答里全是原始 chunk 内容，说明链路是通的，缺的只是一个真模型。
    """

    @property
    def model_name(self) -> str:
        return "mock-echo"

    def chat(self, messages: list[dict]) -> str:
        user = messages[-1]["content"] if messages else ""
        return (
            "【MockLLM 离线模式】未配置真实 LLM，下面是本次请求实际发送给模型的最后一部分内容，"
            "用于确认 Prompt 拼装和引用编号是否正确：\n\n"
            f"{user[:4000]}"
        )


def get_llm_client(settings: Settings) -> BaseLLM:
    """工厂函数：根据 .env 里的 LLM_PROVIDER 决定用哪个实现。"""
    if settings.llm_provider == "openai":
        return OpenAICompatibleLLM(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
        )
    if settings.llm_provider == "anthropic":
        return AnthropicCompatibleLLM(
            model=settings.llm_model or None,
            max_tokens=settings.llm_max_tokens,
        )
    return MockLLM()
