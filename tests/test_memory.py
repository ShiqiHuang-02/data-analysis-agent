"""Memory/context budget and reflection-on-error behaviour."""
from types import SimpleNamespace
import json

import pandas as pd
import pytest

from core.agent import run_agent, REFLECTION_PROMPT
from core.dataset import Dataset


class ScriptedClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return next(self.responses)


def response(tool=None, arguments=None, usage=None):
    calls = [SimpleNamespace(id="call", function=SimpleNamespace(
        name=tool, arguments=json.dumps(arguments)))] if tool else None
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content=None if tool else "分析完成", tool_calls=calls))], usage=usage)


@pytest.fixture
def ds():
    return Dataset(pd.DataFrame({"地区": ["华北", "华南"], "销售额": [100.0, 200.0]}), "sales.csv")


def test_reflection_after_repeated_tool_errors(ds):
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=3, total_tokens=13)
    client = ScriptedClient([
        response("aggregate_table", {"metrics": [{"name": "x", "column": "错列", "agg": "sum"}]}, usage),
        response("aggregate_table", {"metrics": [{"name": "x", "column": "错列2", "agg": "sum"}]}, usage),
        response("aggregate_table", {"metrics": [{"name": "x", "column": "销售额", "agg": "sum"}]}, usage),
        response(usage=usage),
    ])
    result = run_agent("统计总销售额", ds, client=client)
    assert result.stats["reflections"] == 1
    assert result.stats["tool_errors"] == 2
    # 第 3 次请求的消息里应已注入反思提示
    assert any(REFLECTION_PROMPT in m["content"]
               for m in client.calls[2]["messages"] if m.get("role") == "user")
    # 反思后模型修正并成功
    assert result.trace[-1]["result"]["status"] == "ok"


def test_history_budget_keeps_recent_turns(ds, monkeypatch):
    monkeypatch.setattr("core.agent.MAX_HISTORY_CHARS", 100)
    history = [
        {"role": "user", "content": "旧问题" * 20},
        {"role": "assistant", "content": "旧回答" * 20},
        {"role": "user", "content": "最近的问题"},
    ]
    client = ScriptedClient([response("overview", {}, None), response()])
    run_agent("概况", ds, history=history, client=client, first_tool="overview")
    contents = [m["content"] for m in client.calls[0]["messages"] if m["role"] == "user"]
    assert any("最近的问题" in c for c in contents)
    assert all("旧问题" not in c for c in contents)
