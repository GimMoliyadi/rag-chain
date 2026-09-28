"""测试共享的假对象。tests 目录没有 __init__.py，pytest 会把本目录加进 sys.path，
各测试文件直接 `from fakes import ...` 即可。"""
from __future__ import annotations

from app.llm import BaseLLM


class ScriptedLLM(BaseLLM):
    """按脚本依序返回预设文本的假 LLM，并记录每次收到的完整 messages。

    脚本耗尽后再被调用就抛 AssertionError——这本身就是断言手段：
    「这一步不该发生 LLM 调用」的测试让脚本保持空即可。
    """

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[list[dict]] = []

    @property
    def model_name(self) -> str:
        return "scripted"

    def chat(self, messages: list[dict]) -> str:
        self.calls.append(messages)
        if not self._responses:
            raise AssertionError("ScriptedLLM 脚本耗尽：出现了未预期的 LLM 调用")
        return self._responses.pop(0)
