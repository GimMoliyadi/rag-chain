"""LLM provider 的配置校验与 wire 格式适配测试。

前面几层（embedding / retrieval / rag）都已经有测试，但 app/llm.py 一直是盲区：
provider 校验只看 mock 默认值，三个实现类从来没被单测过。这次的改动是给
anthropic 补三项校验（token 存在性 / 模型 ID 非空 / BASE_URL 格式），
所以必须把这些分支锁住，否则下次改配置解析很容易静默改坏。

两条纪律贯穿整个文件：
    1. 绝不依赖真实进程环境，也不依赖真实 .env。两边在开发机上都是真实存在的
       （ANTHROPIC_* 在 Claude Code 进程里，LLM_* 在项目 .env 里），所以每个测试
       都用 monkeypatch 控制环境变量、用 anthropic_settings() 把 .env 的三个字段
       显式压成空串，否则同一段测试在有配置和没配置的机器上结果会不一样。
    2. 不发任何真实网络请求。构造 AnthropicCompatibleLLM 本身不联网，
       但 chat() 会，所以 chat() 的测试把底层 client 换成一个记录调用的假对象。
"""
from __future__ import annotations

import pytest

from app.config import Settings
from app.llm import (
    AnthropicCompatibleLLM,
    MockLLM,
    get_llm_client,
)


# 一个刻意"看起来像密钥"的值，专门用来验证报错信息有没有把它回显出来。
# 真实 token 不进测试代码，也不进任何输出。
FAKE_TOKEN = "sk-test-do-not-log-9f3a1c"


def anthropic_settings(**overrides) -> Settings:
    """一份 anthropic 模式的配置，路径全部指向 tmp 无关的值（这里只用它做校验，不碰磁盘）。

    llm_model / llm_base_url / llm_api_key 都显式传空串，是为了不让它们从真实 .env
    里继承值——否则"模型 ID 为空""凭证为空"这类测试在 .env 填了值的机器上会莫名通过。
    三个都传是为了让"进程环境兜底"这条路径在测试里始终干净可预测。
    """
    kwargs = dict(
        llm_provider="anthropic",
        llm_model="",
        llm_base_url="",
        llm_api_key="",
        embedding_provider="local",
    )
    kwargs.update(overrides)
    return Settings(**kwargs)


# ---- 1. validate_providers 的 anthropic 分支 ----


def test_validate_anthropic_requires_token(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_AUTH_TOKEN"):
        anthropic_settings().validate_providers()


def test_validate_anthropic_rejects_missing_model(monkeypatch):
    """LLM_MODEL 和 ANTHROPIC_MODEL 都空时必须报错，而不是让 SDK 去抛"缺少 model 参数"。"""
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", FAKE_TOKEN)
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    with pytest.raises(ValueError, match="模型 ID 为空"):
        anthropic_settings().validate_providers()


def test_validate_anthropic_accepts_model_from_process_env(monkeypatch):
    """模型 ID 只在进程环境里有，也是合法的（这正是复用 Claude Code 环境的实际路径）。"""
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", FAKE_TOKEN)
    monkeypatch.setenv("ANTHROPIC_MODEL", "sensenova-6.8-flash-lite")
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)  # 留空合法，走 SDK 默认端点
    anthropic_settings().validate_providers()  # 不抛即通过


def test_validate_anthropic_accepts_model_from_env_file(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", FAKE_TOKEN)
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    anthropic_settings(llm_model="sensenova-6.8-flash-lite").validate_providers()


@pytest.mark.parametrize(
    "bad_url",
    [
        "token.sensenova.cn",          # 最常见的错法：漏了 scheme
        "token.sensenova.cn:443",      # 只有 host:port，同样缺 scheme
        "ftp://token.sensenova.cn",    # scheme 不对
        "://token.sensenova.cn",       # scheme 为空
        "http://",                     # 有 scheme 但没 host
    ],
)
def test_validate_anthropic_rejects_bad_base_url(monkeypatch, bad_url):
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", FAKE_TOKEN)
    monkeypatch.setenv("ANTHROPIC_MODEL", "sensenova-6.8-flash-lite")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", bad_url)
    with pytest.raises(ValueError, match="ANTHROPIC_BASE_URL"):
        anthropic_settings().validate_providers()


@pytest.mark.parametrize("good_url", ["https://token.sensenova.cn/compatible-mode/v1", "http://127.0.0.1:8080"])
def test_validate_anthropic_accepts_good_base_url(monkeypatch, good_url):
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", FAKE_TOKEN)
    monkeypatch.setenv("ANTHROPIC_MODEL", "sensenova-6.8-flash-lite")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", good_url)
    anthropic_settings().validate_providers()


def test_validate_error_never_echoes_the_token(monkeypatch):
    """三条报错分支的消息里都不许出现 token 的值——哪怕只是回显前几位。"""
    monkeypatch.setenv("ANTHROPIC_MODEL", "sensenova-6.8-flash-lite")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://token.sensenova.cn")

    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with pytest.raises(ValueError) as exc:
        anthropic_settings().validate_providers()
    assert FAKE_TOKEN not in str(exc.value)

    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", FAKE_TOKEN)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "token.sensenova.cn")
    with pytest.raises(ValueError) as exc:
        anthropic_settings().validate_providers()
    assert FAKE_TOKEN not in str(exc.value)


def test_validate_anthropic_reads_credentials_from_env_file(monkeypatch):
    """.env 的 LLM_API_KEY / LLM_BASE_URL 就能满足校验，进程环境可以完全为空。"""
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    settings = anthropic_settings(
        llm_model="sensenova-6.8-flash-lite",
        llm_base_url="https://token.sensenova.cn/compatible-mode/v1",
        llm_api_key=FAKE_TOKEN,
    )
    settings.validate_providers()


# ---- 2. 三处取值优先级：模型 / 端点 / 凭证共用同一套规则 ----


def test_anthropic_model_prefers_env_file_over_process_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "env-var-model")
    settings = anthropic_settings(llm_model="env-file-model")
    assert settings.anthropic_model == "env-file-model"


def test_anthropic_model_falls_back_to_process_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "env-var-model")
    assert anthropic_settings().anthropic_model == "env-var-model"


def test_anthropic_model_blank_treated_as_missing(monkeypatch):
    """只填了空白字符等于没填，这条分支专门挡"LLM_MODEL=   "这种配置。"""
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    assert anthropic_settings(llm_model="   ").anthropic_model == ""


def test_anthropic_api_key_prefers_env_file_over_process_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "env-var-token")
    assert (
        anthropic_settings(llm_api_key="env-file-token").anthropic_api_key == "env-file-token"
    )


def test_anthropic_api_key_falls_back_to_process_env(monkeypatch):
    """复用 Claude Code 进程环境的实际路径：.env 不填 token，只靠环境变量。"""
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "env-var-token")
    assert anthropic_settings().anthropic_api_key == "env-var-token"


def test_anthropic_api_key_blank_treated_as_missing(monkeypatch):
    """空白 token 不算"填了"，否则校验会放过一个根本认证不了的配置。"""
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "env-var-token")
    assert anthropic_settings(llm_api_key="   ").anthropic_api_key == "env-var-token"


def test_anthropic_base_url_prefers_env_file_over_process_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:8080")
    assert (
        anthropic_settings(llm_base_url="http://127.0.0.1:9090").anthropic_base_url
        == "http://127.0.0.1:9090"
    )


def test_anthropic_base_url_falls_back_to_process_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:8080")
    assert anthropic_settings().anthropic_base_url == "http://127.0.0.1:8080"


# ---- 3. AnthropicCompatibleLLM 构造 ----


def test_anthropic_llm_rejects_empty_model(monkeypatch):
    """空模型必须在构造时就抛，而不是等到第一次 chat() 才发现。"""
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    with pytest.raises(ValueError, match="需要模型 ID"):
        AnthropicCompatibleLLM(model=None)


def test_anthropic_llm_picks_up_model_from_process_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "sensenova-6.8-flash-lite")
    llm = AnthropicCompatibleLLM(model=None)
    assert llm.model_name == "sensenova-6.8-flash-lite"


def test_anthropic_llm_ctor_arg_wins(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "env-var-model")
    assert AnthropicCompatibleLLM(model="ctor-model").model_name == "ctor-model"


def test_anthropic_llm_timeout_reaches_the_sdk(monkeypatch):
    """超时覆盖是这次改动的重点：SDK 默认 connect=5.0s，必须确认我们的值真的传进去了。"""
    monkeypatch.setenv("ANTHROPIC_MODEL", "sensenova-6.8-flash-lite")
    llm = AnthropicCompatibleLLM(timeout_connect=15.0, timeout_total=300.0)
    timeout = llm._client.timeout
    assert (timeout.connect, timeout.read) == (15.0, 300.0)


def test_anthropic_llm_timeout_defaults_are_not_the_sdk_defaults(monkeypatch):
    """两个超时是刻意覆盖的（见 app/config.py 里 llm_timeout_* 的注释）：
    如果被改回 SDK 的 5.0s / 600s，项目就失去对这两个值的独立控制。"""
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    llm = AnthropicCompatibleLLM(model="sensenova-6.8-flash-lite")
    timeout = llm._client.timeout
    assert timeout.connect == 15.0
    assert timeout.read == 300.0


def test_anthropic_llm_uses_credentials_from_ctor_args(monkeypatch):
    """.env 的 LLM_BASE_URL / LLM_API_KEY 会真的传到 SDK，不只是存在 Settings 里。"""
    monkeypatch.setenv("ANTHROPIC_MODEL", "sensenova-6.8-flash-lite")
    llm = AnthropicCompatibleLLM(
        base_url="https://token.sensenova.cn/compatible-mode/v1",
        api_key=FAKE_TOKEN,
    )
    # SDK 会给带 path 的 base_url 补上尾斜杠，断言按归一化后的值写
    assert str(llm._client.base_url) == "https://token.sensenova.cn/compatible-mode/v1/"
    assert llm._client.auth_token == FAKE_TOKEN


def test_anthropic_llm_picks_up_credentials_from_process_env(monkeypatch):
    """只靠进程环境也能构造成功，这是复用 Claude Code 环境的实际路径。"""
    monkeypatch.setenv("ANTHROPIC_MODEL", "sensenova-6.8-flash-lite")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://token.sensenova.cn/compatible-mode/v1")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", FAKE_TOKEN)
    llm = AnthropicCompatibleLLM()
    assert str(llm._client.base_url) == "https://token.sensenova.cn/compatible-mode/v1/"
    assert llm._client.auth_token == FAKE_TOKEN


def test_anthropic_llm_ctor_args_beat_process_env(monkeypatch):
    """两边都填时以构造参数（.env）为准，进程环境不能静默覆盖项目配置。"""
    monkeypatch.setenv("ANTHROPIC_MODEL", "env-var-model")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:8080")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "env-var-token")
    llm = AnthropicCompatibleLLM(
        model="ctor-model",
        base_url="http://127.0.0.1:9090",
        api_key="ctor-token",
    )
    assert llm.model_name == "ctor-model"
    assert str(llm._client.base_url) == "http://127.0.0.1:9090"
    assert llm._client.auth_token == "ctor-token"


def test_anthropic_llm_omits_credentials_when_everything_is_blank(monkeypatch):
    """凭证两处都空时不传参数，让 SDK 走它自己的默认行为（而不是塞个空串进去）。"""
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    llm = AnthropicCompatibleLLM(model="sensenova-6.8-flash-lite")
    # 默认端点 SDK 原样返回；只有"带 path 的自定义 URL"才会被补尾斜杠
    assert str(llm._client.base_url) == "https://api.anthropic.com"
    assert llm._client.auth_token is None


# ---- 4. chat() 的 wire 格式适配 ----


class FakeMessage:
    """SDK 返回的 Message 长这样：chat() 只读它的 .content。"""

    def __init__(self, blocks):
        self.content = blocks


class FakeMessages:
    """记录 create() 收到什么参数，返回一个装着可控 blocks 的 Message。"""

    def __init__(self, blocks):
        self.kwargs = None
        self._blocks = blocks

    def create(self, **kwargs):
        self.kwargs = kwargs
        return FakeMessage(self._blocks)


class FakeBlock:
    """只有 chat() 用到的字段：type 决定它会不会被挑出来。"""

    def __init__(self, type_, text=""):
        self.type = type_
        self.text = text


def _chat_with(fake_blocks, messages, llm=None):
    """把底层 client 换成假对象后跑一次 chat()，返回 create() 收到的参数。"""
    fake = FakeMessages(fake_blocks)
    llm = llm or AnthropicCompatibleLLM(model="sensenova-6.8-flash-lite")
    llm._client = type("Client", (), {"messages": fake})()
    answer = llm.chat(messages)
    return answer, fake.kwargs


def test_chat_moves_system_out_of_messages():
    """Anthropic 不允许 messages 里出现 role=system，这是这个类存在的根本理由。"""
    answer, kwargs = _chat_with(
        [FakeBlock("text", "答案是 X")],
        [{"role": "system", "content": "你是助手"}, {"role": "user", "content": "问题"}],
    )
    assert kwargs["system"] == "你是助手"
    assert kwargs["messages"] == [{"role": "user", "content": "问题"}]
    assert kwargs["model"] == "sensenova-6.8-flash-lite"
    assert kwargs["max_tokens"] == 4096
    assert answer == "答案是 X"


def test_chat_joins_multiple_system_messages():
    answer, kwargs = _chat_with(
        [FakeBlock("text", "ok")],
        [{"role": "system", "content": "规则 A"}, {"role": "system", "content": "规则 B"}],
    )
    assert kwargs["system"] == "规则 A\n\n规则 B"


def test_chat_omits_system_when_absent():
    """没有 system 时不传这个参数：第三方兼容代理对多余参数可能直接 400。"""
    _, kwargs = _chat_with([FakeBlock("text", "ok")], [{"role": "user", "content": "问题"}])
    assert "system" not in kwargs


def test_chat_ignores_non_text_blocks():
    """content 里会混排 thinking / tool_use 块，只取 type == 'text' 的那部分。"""
    answer, kwargs = _chat_with(
        [FakeBlock("thinking", "内部推理"), FakeBlock("text", "第一句"), FakeBlock("tool_use"), FakeBlock("text", "第二句")],
        [{"role": "user", "content": "问题"}],
    )
    assert answer == "第一句第二句"
    assert kwargs["model"] == "sensenova-6.8-flash-lite"


def test_chat_max_tokens_is_configurable():
    _, kwargs = _chat_with(
        [FakeBlock("text", "ok")],
        [{"role": "user", "content": "问题"}],
        llm=AnthropicCompatibleLLM(model="sensenova-6.8-flash-lite", max_tokens=1024),
    )
    assert kwargs["max_tokens"] == 1024


# ---- 5. 工厂路由：三个 provider 共存 ----


def test_factory_routes_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "sensenova-6.8-flash-lite")
    settings = anthropic_settings(llm_model="", llm_timeout_connect=15.0, llm_timeout_total=300.0)
    llm = get_llm_client(settings)
    assert isinstance(llm, AnthropicCompatibleLLM)
    assert llm.model_name == "sensenova-6.8-flash-lite"
    timeout = llm._client.timeout
    assert (timeout.connect, timeout.read) == (15.0, 300.0)


def test_factory_env_file_model_wins_over_process_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "env-var-model")
    settings = anthropic_settings(llm_model="env-file-model")
    assert get_llm_client(settings).model_name == "env-file-model"


def test_factory_env_file_model_blank_falls_back(monkeypatch):
    """.env 里 LLM_MODEL= 是空串（不是 None），必须回退到进程环境的 ANTHROPIC_MODEL。"""
    monkeypatch.setenv("ANTHROPIC_MODEL", "env-var-model")
    settings = anthropic_settings(llm_model="")
    assert get_llm_client(settings).model_name == "env-var-model"


def test_factory_passes_env_file_credentials_to_the_sdk(monkeypatch):
    """.env 里的凭证必须一路传到 SDK；进程环境清空来证明不是环境变量兜底生效的。"""
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    settings = anthropic_settings(
        llm_model="sensenova-6.8-flash-lite",
        llm_base_url="http://127.0.0.1:9090",
        llm_api_key=FAKE_TOKEN,
    )
    llm = get_llm_client(settings)
    assert isinstance(llm, AnthropicCompatibleLLM)
    assert str(llm._client.base_url) == "http://127.0.0.1:9090"
    assert llm._client.auth_token == FAKE_TOKEN


def test_factory_still_returns_mock_for_other_providers(monkeypatch):
    """加了 anthropic 分支之后，mock 的默认路由不能被动到。"""
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    llm = get_llm_client(Settings(llm_provider="mock", llm_model=""))
    assert isinstance(llm, MockLLM)
    assert llm.model_name == "mock-echo"


def test_factory_rejects_anthropic_without_any_model(monkeypatch):
    """工厂路径也要能挡住空模型，不能只在 validate_providers 里挡。"""
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    with pytest.raises(ValueError, match="需要模型 ID"):
        get_llm_client(anthropic_settings(llm_model=""))
