"""Observability helpers: cost estimation and trace export in standard formats.

No third-party SDK is required: the exporters return plain dicts that follow
Langfuse / OpenTelemetry conventions, so a production deployment can forward
them unchanged. The JSON download in the UI is the lightweight path; these
functions are the "how it maps to real observability backends" answer.
"""
from __future__ import annotations

from typing import Any

# 单位：元 / 百万 tokens（输入为缓存未命中价）。价格变动频繁，运行前请核对：
# https://api-docs.deepseek.com/quick_start/pricing
PRICE_CNY_PER_1M: dict[str, dict[str, float]] = {
    "deepseek-chat": {"input": 2.0, "output": 8.0},
    "deepseek-reasoner": {"input": 4.0, "output": 16.0},
    "deepseek-flash": {"input": 1.0, "output": 2.0},
    "deepseek-v4-flash": {"input": 1.0, "output": 2.0},
    "deepseek-v4-pro": {"input": 3.0, "output": 6.0},
}


def estimate_cost(model: str | None, prompt_tokens: int | None,
                  completion_tokens: int | None) -> float | None:
    """按配置单价估算单次请求成本（元）。Token 未返回或模型未知时返回 None。"""
    if prompt_tokens is None or completion_tokens is None or model is None:
        return None
    price = PRICE_CNY_PER_1M.get(model)
    if price is None:
        return None
    return (prompt_tokens * price["input"] + completion_tokens * price["output"]) / 1_000_000


def _usage(stats: dict) -> dict[str, Any]:
    return {
        "input": stats.get("prompt_tokens"),
        "output": stats.get("completion_tokens"),
        "total": stats.get("total_tokens"),
    }


def to_langfuse_trace(result: Any) -> dict:
    """把 AgentResult 转成 Langfuse 风格 trace：一次 generation + N 个 tool span。"""
    stats = result.stats
    observations: list[dict] = [{
        "type": "generation",
        "name": "agent-response",
        "model": stats.get("model"),
        "output": result.answer,
        "usage": _usage(stats),
        "metadata": {
            "finish_reason": stats.get("finish_reason"),
            "elapsed_ms": stats.get("elapsed_ms"),
            "model_requests": stats.get("model_requests"),
            "tools_executed": stats.get("tools_executed"),
            "tool_errors": stats.get("tool_errors"),
            "estimated_cost_cny": stats.get("estimated_cost_cny"),
        },
    }]
    for entry in result.trace:
        observations.append({
            "type": "span",
            "name": f"tool:{entry['tool']}",
            "input": {"tool": entry["tool"], "arguments": entry.get("arguments")},
            "output": entry.get("result"),
            "duration_ms": entry.get("duration_ms"),
        })
    return {
        "trace": {
            "name": "data-analysis-agent",
            "metadata": {"model": stats.get("model"), "profile": stats.get("profile")},
            "observations": observations,
        },
    }


def to_otlp_spans(result: Any) -> list[dict]:
    """把 AgentResult 转成 OpenTelemetry span 结构（gen_ai 语义约定）。"""
    stats = result.stats
    spans: list[dict] = [{
        "name": "agent.turn",
        "attributes": {
            "gen_ai.model": stats.get("model"),
            "gen_ai.finish_reason": stats.get("finish_reason"),
            "gen_ai.usage.prompt_tokens": stats.get("prompt_tokens"),
            "gen_ai.usage.completion_tokens": stats.get("completion_tokens"),
            "gen_ai.usage.total_tokens": stats.get("total_tokens"),
            "gen_ai.estimated_cost_cny": stats.get("estimated_cost_cny"),
            "agent.model_requests": stats.get("model_requests"),
            "agent.tools_executed": stats.get("tools_executed"),
            "agent.tool_errors": stats.get("tool_errors"),
        },
    }]
    for entry in result.trace:
        spans.append({
            "name": f"tool.{entry['tool']}",
            "attributes": {
                "tool.name": entry["tool"],
                "tool.arguments": entry.get("arguments"),
                "tool.status": (entry.get("result") or {}).get("status"),
                "tool.duration_ms": entry.get("duration_ms"),
            },
        })
    return spans
