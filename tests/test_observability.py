"""Observability: cost estimation and standard trace export."""
from types import SimpleNamespace

import pytest

from core.observability import estimate_cost, to_langfuse_trace, to_otlp_spans


def test_estimate_cost():
    assert estimate_cost("deepseek-chat", 1_000_000, 500_000) == pytest.approx(6.0)
    assert estimate_cost("unknown-model", 100, 100) is None
    assert estimate_cost("deepseek-chat", None, 100) is None


def _result():
    trace = [{"tool": "overview", "arguments": {},
              "result": {"status": "ok", "kind": "overview"}, "duration_ms": 12.0}]
    stats = {"model": "deepseek-chat", "profile": "sales", "prompt_tokens": 10,
             "completion_tokens": 5, "total_tokens": 15, "finish_reason": "completed",
             "elapsed_ms": 100.0, "model_requests": 1, "tools_executed": 1,
             "tool_errors": 0, "estimated_cost_cny": 0.00006}
    return SimpleNamespace(answer="完成", trace=trace, stats=stats)


def test_langfuse_trace_shape():
    t = to_langfuse_trace(_result())
    assert t["trace"]["name"] == "data-analysis-agent"
    obs = t["trace"]["observations"]
    assert obs[0]["type"] == "generation"
    assert obs[1]["type"] == "span" and obs[1]["name"] == "tool:overview"
    assert obs[0]["usage"]["total"] == 15
    assert obs[1]["duration_ms"] == 12.0


def test_otlp_spans_shape():
    spans = to_otlp_spans(_result())
    assert spans[0]["name"] == "agent.turn"
    assert spans[1]["name"] == "tool.overview"
    assert spans[0]["attributes"]["gen_ai.estimated_cost_cny"] == 0.00006
