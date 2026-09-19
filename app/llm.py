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
from collections.abc import Iterator

from .config import Settings


class BaseLLM:
    """所有 LLM 实现的共同接口：收一组 messages，回一段文本。"""

    def chat(self, messages: list[dict]) -> str:
        raise NotImplementedError

    def chat_stream(self, messages: list[dict]) -> Iterator[str]:
        """流式版 chat：逐段产出正文。

        默认实现并不真正流式——一次性调用 chat() 然后整段吐出。这样不支持
        增量输出的实现（MockLLM）不用改就能被流式接口复用；真正支持增量的
        实现各自覆盖这个方法。调用 chat() 发生在第一次 next() 时，
        没人消费这个生成器就不会白跑一次请求。
        """
        yield self.chat(messages)

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

    def _create_kwargs(self, messages: list[dict]) -> dict:
        """chat() 和 chat_stream() 共用的请求参数，避免两处各写一遍。"""
        return {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
        }

    def chat(self, messages: list[dict]) -> str:
        response = self._client.chat.completions.create(**self._create_kwargs(messages))
        return response.choices[0].message.content or ""

    def chat_stream(self, messages: list[dict]) -> Iterator[str]:
        with self._client.chat.completions.create(
            **self._create_kwargs(messages), stream=True
        ) as stream:
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta


class AnthropicCompatibleLLM(BaseLLM):
    """Anthropic Messages API 端点：官方 API 或 CC Switch 这类兼容代理。

    和 OpenAICompatibleLLM 的差别不在模型，而在 wire 格式。Anthropic 不允许
    messages 数组里出现 role="system"——system 必须是请求的顶层参数，而且首条
    message 必须是 user。本项目 rag.py 拼出来的是 OpenAI 风格的 [system, user]，
    所以 chat() 先在边界上把 system 条目摘出去再交给 SDK。适配只发生在这一个
    类里，rag.py 完全不用改。

    模型 / 端点 / 凭证的优先级统一为：构造参数（来自 .env 的 LLM_MODEL /
    LLM_BASE_URL / LLM_API_KEY）优先，进程环境的 ANTHROPIC_MODEL /
    ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN 兜底。这样既能只靠 .env 独立运行
    （在 Claude Code 之外的普通终端里也有用），又能继续复用 Claude Code 的进程
    环境。token 仍然不进代码，只会落在被 gitignore 的 .env 里。

    不传 temperature / thinking 这类参数：本端点是第三方兼容代理，只验证过最简
    请求（model + max_tokens + messages），加 Anthropic 专有参数很可能直接 400。
    少传参数就是少一个失败点。

    超时是刻意覆盖的（见 config.py 里 llm_timeout_* 的注释）：connect 是为了
    连不上时快速失败，total 是给等回答的耗时留余量。上游 LLM 延迟受服务负载、
    网络和时段影响较大，历史实测数字已经证明不稳定，所以这里不写具体测量值——
    需要新数字时跑 `python scripts/probe_llm_latency.py`。两个值都可用 .env 配置。

    模型 ID 为空时直接在这里抛，而不是让 SDK 返回一个"缺少 model 参数"的错：
    后者看不出是配置问题，排查要多绕一层。
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 4096,
        timeout_connect: float = 15.0,
        timeout_total: float = 300.0,
    ) -> None:
        import anthropic  # 延迟导入，mock / openai 模式下不需要 anthropic
        from httpx2 import Timeout  # anthropic 自身依赖，不会引入新包

        # 三处遵循同一条优先级：构造参数（.env 显式配置）优先，进程环境兜底。
        self._model = (model or os.environ.get("ANTHROPIC_MODEL", "")).strip()
        if not self._model:
            raise ValueError(
                "AnthropicCompatibleLLM 需要模型 ID：构造参数或 ANTHROPIC_MODEL 都没有。"
                "请在 .env 的 LLM_MODEL 或进程环境的 ANTHROPIC_MODEL 里填一处。"
            )
        self._base_url = (base_url or os.environ.get("ANTHROPIC_BASE_URL", "")).strip()
        self._auth_token = (api_key or os.environ.get("ANTHROPIC_AUTH_TOKEN", "")).strip()

        # 留空就不传：缺 base_url 时 SDK 走 api.anthropic.com，缺 auth_token 时
        # SDK 自己读 ANTHROPIC_AUTH_TOKEN。构造本身不校验凭证，因为 SDK 只在
        # 第一次请求时才真正校验——把校验前置在 config.validate_providers()。
        client_kwargs: dict[str, object] = {
            "timeout": Timeout(timeout=timeout_total, connect=timeout_connect)
        }
        if self._base_url:
            client_kwargs["base_url"] = self._base_url
        if self._auth_token:
            client_kwargs["auth_token"] = self._auth_token
        self._client = anthropic.Anthropic(**client_kwargs)
        self._max_tokens = max_tokens

    @property
    def model_name(self) -> str:
        return self._model

    def _split_system(self, messages: list[dict]) -> tuple[str, list[dict]]:
        """拆 system：Anthropic 要求它作顶层参数，messages 里只留 user / assistant。"""
        system_parts: list[str] = []
        api_messages: list[dict] = []
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                api_messages.append({"role": msg["role"], "content": msg["content"]})
        return "\n\n".join(system_parts), api_messages

    def _create_kwargs(self, messages: list[dict]) -> dict[str, object]:
        """chat() 和 chat_stream() 共用的请求参数，两处不能各写一份。"""
        system_text, api_messages = self._split_system(messages)
        create_kwargs: dict[str, object] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": api_messages,
        }
        if system_text:
            create_kwargs["system"] = system_text
        return create_kwargs

    def chat(self, messages: list[dict]) -> str:
        response = self._client.messages.create(**self._create_kwargs(messages))
        # 响应的 content 是 block 列表，text / thinking / tool_use 混排，
        # 必须按 type 挑，不能假设第一个块就是文本。
        return "".join(block.text for block in response.content if block.type == "text")

    def chat_stream(self, messages: list[dict]) -> Iterator[str]:
        """流式版：只吐 text_delta 的正文。

        实测这个端点的正文在 content_block_delta / text_delta 里（字段 text），
        thinking_delta 不带正文，统一跳过。
        """
        with self._client.messages.create(
            **self._create_kwargs(messages), stream=True
        ) as stream:
            for event in stream:
                if event.type != "content_block_delta":
                    continue
                if event.delta.type == "text_delta" and event.delta.text:
                    yield event.delta.text


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
        # 三个 anthropic_* 属性在这里统一取值（.env 优先、进程环境兜底），
        # 优先级只在这一处解析，实现类不用再各自重复一遍。
        return AnthropicCompatibleLLM(
            model=settings.anthropic_model or None,
            base_url=settings.anthropic_base_url or None,
            api_key=settings.anthropic_api_key or None,
            max_tokens=settings.llm_max_tokens,
            timeout_connect=settings.llm_timeout_connect,
            timeout_total=settings.llm_timeout_total,
        )
    return MockLLM()
