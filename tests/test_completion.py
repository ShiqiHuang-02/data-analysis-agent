"""Regression tests for the final resume-project feature gaps."""
from io import BytesIO
from types import SimpleNamespace
import json

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from core.agent import run_agent
from core.cleaning import proposal_from_trace, confirm_proposal
from core.dataset import Dataset
from core.schemas import function_specs
from core.tools import run_tool
from core.visualization import build_chart


@pytest.fixture
def dated():
    # Deliberately out-of-order dates, a malformed date, and a missing metric.
    return Dataset(pd.DataFrame({
        "日期": ["2026-11-10", "2026/09/01", "bad", "2026-11-02", "2026-10-03", "2026-10-15"],
        "门店": ["甲", "甲", "乙", "乙", "甲", "乙"],
        "销售额": [10.0, 30.0, 10000.0, 40.0, 20.0, None],
    }), "sales.csv")


def test_monthly_trend_correct_chronology_and_invalid_dates(dated):
    r = run_tool(dated, "time_trend", {"date_col": "日期", "value_col": "销售额", "frequency": "month", "agg": "sum"})
    assert r["status"] == "ok"
    assert r["invalid_date_rows"] == 1
    assert r["valid_date_rows"] == 5
    assert r["total_periods"] == 3
    assert r["metric"] == "销售额_sum"
    assert r["table"] == [
        {"period": "2026-09-01", "销售额_sum": 30.0},
        {"period": "2026-10-01", "销售额_sum": 20.0},
        {"period": "2026-11-01", "销售额_sum": 50.0},
    ]
    assert r["note"].startswith("只展示实际出现")


def test_trend_count_and_recent_periods(dated):
    r = run_tool(dated, "time_trend", {"date_col": "日期", "frequency": "month", "agg": "count", "max_periods": 2})
    assert r["status"] == "ok"
    assert r["table"] == [
        {"period": "2026-10-01", "count": 2},
        {"period": "2026-11-01", "count": 2},
    ]
    assert r["truncated"] is True
    assert r["total_periods"] == 3


def test_trend_filters_and_weekly_granularity(dated):
    r = run_tool(dated, "time_trend", {
        "date_col": "日期", "frequency": "week", "value_col": "销售额", "agg": "mean",
        "where": {"column": "门店", "operator": "eq", "value": "乙"},
    })
    assert r["status"] == "ok"
    assert r["filtered_rows"] == 3
    assert r["invalid_date_rows"] == 1
    assert r["table"] == [{"period": "2026-10-12", "销售额_mean": None},
                          {"period": "2026-11-02", "销售额_mean": 40.0}]


def test_bad_date_type_and_invalid_date_not_silently_plotted(dated):
    assert run_tool(dated, "time_trend", {"date_col": "日期", "value_col": "销售额", "agg": "sum", "frequency": "day", "max_periods": 0})["status"] == "error"
    assert run_tool(dated, "time_trend", {"date_col": "销售额", "frequency": "day"})["status"] == "error"
    bad = Dataset(pd.DataFrame({"date": ["not date", "still not date"]}))
    assert run_tool(bad, "time_trend", {"date_col": "date"})["status"] == "error"


def test_time_trend_renders_png(dated):
    r = run_tool(dated, "time_trend", {"date_col": "日期", "value_col": "销售额", "agg": "sum", "frequency": "month"})
    fig = build_chart(dated, r["chart_spec"])
    output = BytesIO()
    fig.savefig(output, format="png")
    plt.close(fig)
    assert output.getvalue().startswith(b"\x89PNG")


def test_no_trend_invented_zeros(dated):
    r = run_tool(dated, "time_trend", {"date_col": "日期", "frequency": "day"})
    assert len(r["table"]) == 5  # only observed valid days
    assert not any(row["period"] == "2026-11-09" for row in r["table"])


def test_agent_clean_proposal_then_confirm_is_explicit(dated):
    original = dated.df.copy(deep=True)
    trace = [{"tool": "clean_preview", "arguments": {"operation": "drop_missing", "column": "销售额"},
              "result": run_tool(dated, "clean_preview", {"operation": "drop_missing", "column": "销售额"})}]
    proposal = proposal_from_trace(trace, revision=2)
    assert proposal is not None
    pd.testing.assert_frame_equal(original, dated.df)
    with pytest.raises(ValueError, match="重新生成"):
        confirm_proposal(dated, proposal, revision=3)
    updated = confirm_proposal(dated, proposal, revision=2)
    assert len(updated.df) == len(original)-1
    assert len(dated.df) == len(original)


def test_forged_proposal_is_rejected(dated):
    clean_args = {"operation": "drop_missing", "column": "销售额"}
    p = {"revision": 1, "arguments": clean_args,
         "preview": {**run_tool(dated, "clean_preview", clean_args), "after_rows": 0}}
    with pytest.raises(ValueError, match="预览时不同"):
        confirm_proposal(dated, p, revision=1)
    assert proposal_from_trace([{"tool": "clean_preview", "arguments": {"operation":"bad"},
                                 "result": {"status": "error"}}], revision=1) is None


class StubClient:
    def __init__(self, tool_name, args):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.calls = []
        self.tool_name = tool_name
        self.args = args

    def create(self, **kw):
        self.calls.append(kw)
        if len(self.calls) == 1:
            msg = SimpleNamespace(content=None, tool_calls=[SimpleNamespace(
                id="call1", function=SimpleNamespace(name=self.tool_name,
                                                      arguments=json.dumps(self.args, ensure_ascii=False)))])
        else:
            msg = SimpleNamespace(content="工具执行完成。", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def test_agent_discovers_trend_and_first_tool_is_enforced(dated):
    client = StubClient("time_trend", {"date_col": "日期", "frequency": "month", "agg": "count"})
    response = run_agent("按月统计", dated, client=client, first_tool="time_trend")
    assert response.trace[0]["tool"] == "time_trend"
    assert response.trace[0]["result"]["status"] == "ok"
    assert client.calls[0]["tool_choice"]["function"]["name"] == "time_trend"
    assert client.calls[1]["tool_choice"] == "auto"
    assert "time_trend" in [s["function"]["name"] for s in function_specs()]
    with pytest.raises(ValueError):
        run_agent("任意执行", dated, client=client, first_tool="execute_pandas_code")


def test_agent_clean_response_creates_pending_proposal(dated):
    client = StubClient("clean_preview", {"operation": "drop_duplicates"})
    reply = run_agent("清理重复行", dated, client=client)
    proposal = proposal_from_trace(reply.trace, revision=0)
    assert proposal["preview"]["operation"] == "drop_duplicates"
    pd.testing.assert_frame_equal(confirm_proposal(dated, proposal, revision=0).df, dated.df.drop_duplicates())
