"""Self-checks for the eval grader: canonicalization, equivalence, cost, mock run."""
import pytest

from core.dataset import Dataset
from core.tools import run_tool
from evals.runner import (
    canonicalize,
    estimate_cost,
    results_match,
    aggregate,
    run_mock,
    hallucinated_numbers,
)
from evals.tasks import TASKS


@pytest.fixture
def ds():
    import pandas as pd
    return Dataset(pd.DataFrame({
        "日期": ["2026-01-01", "2026-01-15", "2026-02-01", "2026-02-20",
                 "2026-03-01", "2026-03-15", "2026-04-01", "2026-04-20"],
        "地区": ["华北", "华北", "华南", "华南", "华东", "西南", "华北", "华南"],
        "渠道": ["官网", "官网", "门店", "门店", "电商", "官网", "电商", "门店"],
        "订单数": [3, 4, 5, 6, 7, 8, 9, 10],
        "销售额": [100.0, 200.0, 300.0, 400.0, None, 600.0, 700.0, 800.0],
        "客户满意度": [4.0, None, 5.0, 3.0, 5.0, None, 4.0, 5.0],
        "是否复购": ["否", "是", "是", "否", "是", "否", "是", "是"],
    }), "sales.csv")


def test_canonicalize_sorts_rows_and_rounds_floats():
    a = {"status": "ok", "kind": "table",
         "table": [{"group": "B", "v": 1.00001}, {"group": "A", "v": 2.0}]}
    b = {"status": "ok", "kind": "table",
         "table": [{"v": 2.0, "group": "A"}, {"v": 1.00001, "group": "B"}]}
    assert canonicalize(a, "group_aggregate") == canonicalize(b, "group_aggregate")


def test_aggregate_table_ignores_metric_names(ds):
    golden = run_tool(ds, "aggregate_table", {
        "group_by": ["地区"],
        "metrics": [{"name": "金额", "column": "销售额", "agg": "sum"},
                    {"name": "行数", "agg": "row_count"}]})
    agent = run_tool(ds, "aggregate_table", {
        "group_by": ["地区"],
        "metrics": [{"name": "sales_sum", "column": "销售额", "agg": "sum"},
                    {"name": "rows", "agg": "row_count"}]})
    assert results_match(golden, agent, "aggregate_table", ["地区"])


def test_results_match_detects_wrong_values(ds):
    golden = run_tool(ds, "group_aggregate", {"group_by": "地区", "value_col": "销售额", "agg": "sum"})
    wrong = run_tool(ds, "group_aggregate", {"group_by": "地区", "value_col": "订单数", "agg": "sum"})
    assert not results_match(golden, wrong, "group_aggregate")


def test_estimate_cost():
    stats = {"prompt_tokens": 1_000_000, "completion_tokens": 500_000}
    assert estimate_cost("deepseek-chat", stats) == pytest.approx(6.0)  # 1M*2 + 0.5M*8
    assert estimate_cost("unknown-model", stats) is None
    assert estimate_cost("deepseek-chat", {"prompt_tokens": None, "completion_tokens": None}) is None


def test_hallucinated_numbers_flags_absent_numbers():
    trace = [{"result": {"status": "ok", "kind": "table", "table": [{"v": 1234.5}]}}]
    assert 999.0 in hallucinated_numbers("结果是 999 和 1234.5", trace)
    assert 1234.5 not in hallucinated_numbers("结果是 1234.5", trace)


def test_mock_self_check_is_perfect(ds):
    # 注入黄金工具调用后，评分器应给全部分数：既验证评分器正路径，也验证
    # 全部任务集的 golden_args 在这份数据上都能成功执行（status == ok）。
    scores = run_mock(TASKS, ds)
    agg = aggregate(scores)
    assert agg["routing_accuracy"] == 1.0
    assert agg["result_accuracy"] == 1.0
    assert agg["completion_rate"] == 1.0
