"""Business-level results and agent feedback/observability regression tests."""
from types import SimpleNamespace
import json

import pandas as pd
import pytest

from core.agent import run_agent
from core.dataset import Dataset
from core.profiles import resolve_profile
from core.tools import run_tool


@pytest.fixture
def sales():
    return Dataset(pd.DataFrame({
        "地区": ["华北", "华北", "华北", "华南", "华南", None],
        "渠道": ["官网", "官网", "门店", "官网", "门店", "官网"],
        "订单号": ["A", "A", "B", "C", None, "D"],
        "订单数": [2, 3, 1, 4, 5, 2],
        "销售额": [100.0, None, 200.0, None, None, 50.0],
        "日期": ["2026-01-01", "2026/01/02", "bad", "2026-02-01", None, "2026-03-01"],
    }), "sales.csv")


def summarize(ds, **extra):
    args = {"group_by": ["地区"], "metrics": [{"name": "金额", "column": "销售额", "agg": "sum"}]}
    return run_tool(ds, "aggregate_table", {**args, **extra})


def test_multiple_dimensions_metrics_and_missing_groups(sales):
    result = summarize(sales, group_by=["地区", "渠道"], metrics=[
        {"name": "金额", "column": "销售额", "agg": "sum"},
        {"name": "行数", "agg": "row_count"},
        {"name": "有效金额数", "column": "销售额", "agg": "count"},
        {"name": "订单去重数", "column": "订单号", "agg": "nunique"},
        {"name": "订单量", "column": "订单数", "agg": "sum"},
    ])
    assert result["status"] == "ok"
    north = next(row for row in result["table"] if row["地区"] == "华北" and row["渠道"] == "官网")
    assert north == {"地区": "华北", "渠道": "官网", "金额": 100.0, "行数": 2,
                     "有效金额数": 1, "订单去重数": 1, "订单量": 5}
    assert all(row["金额"] is None for row in result["table"] if row["地区"] == "华南")
    assert any(row["地区"] is None and row["金额"] == 50 for row in result["table"])
    assert result["metric_definitions"][0]["valid_rows"] == 3
    json.dumps(result, allow_nan=False, ensure_ascii=False)


def test_overall_and_top_n(sales):
    assert summarize(sales, group_by=[])["table"] == [{"金额": 350.0}]
    result = summarize(sales, sort_by="金额", top_n=1)
    assert result["table"] == [{"地区": "华北", "金额": 300.0}]
    assert result["total_groups"] == 3 and result["truncated"]


@pytest.mark.parametrize("logic,expected", [("and", 100.0), ("or", 350.0)])
def test_compound_filters(sales, logic, expected):
    where = {"logic": logic, "conditions": [
        {"column": "地区", "value": "华北"}, {"column": "渠道", "value": "官网"}]}
    result = summarize(sales, group_by=[], where=where)
    assert result["table"] == [{"金额": expected}]
    assert result["applied_filter"]["logic"] == logic


def test_date_filter_and_invalid_rows(sales):
    where = {"logic": "and", "conditions": [
        {"column": "日期", "operator": "ge", "value": "2026-01-01", "value_type": "datetime"},
        {"column": "日期", "operator": "lt", "value": "2026-02-01", "value_type": "datetime"},
    ]}
    result = summarize(sales, where=where)
    assert result["filtered_rows"] == 2
    assert result["table"] == [{"地区": "华北", "金额": 100.0}]
    remaining = sales.filter_df({"column": "日期", "operator": "ne", "value": "2026-01-01", "value_type": "datetime"})
    assert len(remaining) == 3
    assert len(sales.filter_df({"column": "地区", "operator": "ne", "value": "华北"})) == 2


def test_empty_filter_semantics(sales):
    where = {"column": "地区", "value": "不存在"}
    assert summarize(sales, where=where)["table"] == []
    result = summarize(sales, group_by=[], where=where, metrics=[
        {"name": "金额", "column": "销售额", "agg": "sum"}, {"name": "行数", "agg": "row_count"}])
    assert result["table"] == [{"金额": None, "行数": 0}]


@pytest.mark.parametrize("changes", [
    {"metrics": [{"name": "金额", "column": "地区", "agg": "mean"}]},
    {"metrics": [{"name": "地区", "column": "销售额", "agg": "sum"}]},
    {"metrics": [{"name": "x", "agg": "count"}]},
    {"metrics": [{"name": "x", "agg": "row_count", "column": "地区"}]},
    {"metrics": [{"name": "x", "agg": "row_count"}] * 2},
    {"sort_by": "不存在"},
    {"where": {"logic": "and", "conditions": []}},
    {"where": {"column": "日期", "operator": "ge", "value": "bad", "value_type": "datetime"}},
])
def test_invalid_analysis_is_rejected(sales, changes):
    assert summarize(sales, **changes)["status"] == "error"


def test_compound_filter_works_in_trend_and_chart(sales):
    where = {"logic": "and", "conditions": [
        {"column": "地区", "value": "华北"}, {"column": "渠道", "value": "官网"}]}
    trend = run_tool(sales, "time_trend", {"date_col": "日期", "value_col": "销售额", "agg": "sum", "where": where})
    assert trend["table"] == [{"period": "2026-01-01", "销售额_sum": 100.0}]
    chart = run_tool(sales, "make_chart", {"kind": "bar", "x": "地区", "where": where})
    assert chart["filtered_rows"] == 2
    from core.visualization import build_chart
    import matplotlib.pyplot as plt
    figure = build_chart(sales, chart["chart_spec"])
    assert figure.axes[0].patches[0].get_height() == 2
    plt.close(figure)


def test_profiles_are_explicit_and_not_based_on_filename(sales):
    assert resolve_profile(sales)["id"] == "sales"
    assert resolve_profile(sales, "general")["id"] == "general"
    unknown = Dataset(pd.DataFrame({"x": [1]}), "sales.csv")
    assert resolve_profile(unknown)["id"] == "general"
    assert resolve_profile(unknown, "sales")["id"] == "sales"
    with pytest.raises(ValueError):
        resolve_profile(sales, "unknown")


class ScriptedClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def response(tool=None, arguments=None, usage=None):
    calls = [SimpleNamespace(id="call", function=SimpleNamespace(name=tool, arguments=json.dumps(arguments)))] if tool else None
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None if tool else "分析完成", tool_calls=calls))], usage=usage)


def test_usage_and_error_feedback_then_recovery(sales):
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20, total_tokens=120)
    client = ScriptedClient([
        response("aggregate_table", {"metrics": [{"name": "金额", "column": "错列", "agg": "sum"}]}, usage),
        response("aggregate_table", {"metrics": [{"name": "金额", "column": "销售额", "agg": "sum"}]}, usage),
        response(usage=usage),
    ])
    result = run_agent("统计总销售额", sales, client=client)
    assert result.trace[0]["result"]["status"] == "error"
    assert result.trace[1]["result"]["table"] == [{"金额": 350.0}]
    assert result.stats["total_tokens"] == 360
    assert result.stats["usage_status"] == "complete"
    assert result.stats["model_requests"] == 3
    assert result.stats["tools_executed"] == 2
    assert result.stats["tool_errors"] == 1
    assert result.stats["elapsed_ms"] >= result.stats["model_ms"]
    assert all(entry["duration_ms"] >= 0 for entry in result.trace)
    assert "销售分析口径" in client.calls[0]["messages"][0]["content"]


def test_usage_missing_and_failure_partial(sales):
    result = run_agent("你好", sales, client=ScriptedClient([response()]))
    assert result.stats["total_tokens"] is None
    assert result.stats["usage_status"] == "unavailable"
    usage = {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10}
    client = ScriptedClient([response("overview", {}, usage), RuntimeError("private credential")])
    result = run_agent("概况", sales, client=client)
    assert result.stats["total_tokens"] == 10
    assert result.stats["usage_status"] == "partial"
    assert result.stats["finish_reason"] == "model_error"
    assert "private credential" not in result.answer
    assert len(result.trace) == 1


def test_tool_limit_and_statistics(sales):
    item = response("overview", {})
    item.choices[0].message.tool_calls *= 12
    result = run_agent("概况", sales, client=ScriptedClient([item]))
    assert result.stats["tools_executed"] == 10
    assert result.stats["tool_requests"] == 12
    assert result.stats["tool_errors"] == 2
    assert result.stats["finish_reason"] == "limit_reached"
