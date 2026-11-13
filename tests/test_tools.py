"""Deterministic unit and integration tests — no paid API calls."""
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import json
import math
import pandas as pd
import pytest

from core.dataset import Dataset, MAX_UPLOAD_BYTES
from core.tools import run_tool, compute_clean
from core.visualization import build_chart
from core.schemas import function_specs
from core.agent import run_agent


@pytest.fixture
def dataset():
    return Dataset(pd.DataFrame({
        "地区": ["华北", "华北", "华南", "华南", "华南"],
        "销售额": [100, 200, 300, 400, None],
        "客户": ["张一", "张二", "张三", "张四", "张五"],
        "评分": [3.0, 4.0, None, 4.0, 5.0],
    }), "demo.csv")


def test_dataset_read_csv_and_schema_no_raw_values(dataset):
    data = "地区,销售额\n华北,100\n华南,150\n".encode()
    ds = Dataset.from_bytes(data, "demo.csv")
    assert ds.df.shape == (2, 2)
    assert "华北" not in json.dumps(ds.schema(), ensure_ascii=False)
    assert "张一" not in json.dumps(dataset.schema(), ensure_ascii=False)
    assert ds.schema()["row_count"] == 2


def test_xlsx_upload(dataset):
    bio = BytesIO()
    dataset.df.to_excel(bio, index=False)
    parsed = Dataset.from_bytes(bio.getvalue(), "demo.xlsx")
    assert parsed.df.shape == dataset.df.shape


def test_bad_format_and_limit():
    with pytest.raises(ValueError):
        Dataset.from_bytes(b"garbage", "a.exe")
    with pytest.raises(ValueError):
        Dataset.from_bytes(b"x"*(MAX_UPLOAD_BYTES+1), "a.csv")


def test_group_sums(dataset):
    r = run_tool(dataset,"group_aggregate",{"group_by":"地区","value_col":"销售额","agg":"sum"})
    assert r["status"] == "ok"
    assert {x["group"]: x["销售额_sum"] for x in r["table"]} == {"华北": 300.0, "华南": 700.0}


def test_group_row_counts_including_nan(dataset):
    r = run_tool(dataset,"group_aggregate",{"group_by":"地区","agg":"count"})
    assert {x["group"]: x["count"] for x in r["table"]} == {"华北": 2, "华南": 3}


def test_filter_numeric(dataset):
    r = run_tool(dataset, "group_aggregate", {
        "group_by":"地区", "agg":"count",
        "where": {"column":"销售额","operator":"ge","value": 200}})
    assert r["filtered_rows"] == 3
    assert {x["group"]:x["count"] for x in r["table"]} == {"华北":1,"华南":2}


def test_missing_and_corr(dataset):
    r = run_tool(dataset, "missing_values", {})
    assert next(x for x in r["table"] if x["column"] == "评分")["missing"] == 1
    corr = run_tool(dataset, "correlation", {"columns": ["销售额", "评分"]})
    assert corr["status"] == "ok"
    assert corr["table"][0]["销售额"] == 1.0


def test_invalid_and_unknown_tools(dataset):
    assert run_tool(dataset, "execute_pandas_code", {"code":"import os"})["status"] == "error"
    assert run_tool(dataset, "group_aggregate", {"group_by":"地区", "agg":"import os"})["status"] == "error"
    assert run_tool(dataset, "value_counts", {"column":"非现有字段"})["status"] == "error"
    assert run_tool(dataset, "overview", {"unexpected":"inject"})["status"] == "error"
    assert not any(x["function"]["name"] == "execute_pandas_code" for x in function_specs())


def test_clean_preview_is_read_only(dataset):
    before = dataset.df.copy(deep=True)
    r = run_tool(dataset, "clean_preview", {"operation":"fill_median", "column":"评分"})
    assert r["before_missing"] == 2
    assert r["after_missing"] == 1
    pd.testing.assert_frame_equal(dataset.df, before)
    updated = compute_clean(dataset,{"operation":"fill_median", "column":"评分"})
    assert updated["评分"].isna().sum() == 0
    pd.testing.assert_frame_equal(dataset.df, before)


def test_chart_returns_spec_and_renders_locally(dataset):
    spec = {"kind":"bar","x":"地区","y":"销售额","agg":"sum"}
    r=run_tool(dataset,"make_chart",spec)
    assert r["status"] == "ok"
    assert "chart_spec" in r and "raw_rows" not in r
    fig=build_chart(dataset,r["chart_spec"])
    out=BytesIO()
    fig.savefig(out,format="png")
    assert out.getvalue().startswith(b"\x89PNG")
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_bad_column(dataset):
    assert run_tool(dataset, "make_chart", {"kind":"scatter", "x":"地区", "y":"评分"})["status"] == "error"


def test_fixture_csv():
    sample = Path(__file__).resolve().parents[1] / "sample_data" / "sales_demo.csv"
    ds = Dataset.from_bytes(sample.read_bytes(),sample.name)
    assert len(ds.df) == 120
    assert run_tool(ds, "group_aggregate", {"group_by":"地区", "agg":"sum", "value_col":"销售额"})["status"] == "ok"


class FakeCompletion:
    def __init__(self):
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
    def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            tool=SimpleNamespace(id="call_demo", function=SimpleNamespace(name="group_aggregate",arguments=json.dumps({"group_by":"地区","value_col":"销售额","agg":"sum"})))
            msg=SimpleNamespace(content=None,tool_calls=[tool])
        else:
            assert any(m.get("role") == "tool" for m in kwargs["messages"])
            msg=SimpleNamespace(content="华南合计700，华北合计300。",tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def test_agent_tool_round_trip_without_api_key(dataset):
    fake = FakeCompletion()
    res = run_agent("分别求每个地区销售额", dataset, client=fake)
    assert fake.calls == 2
    assert res.trace[0]["result"]["status"] == "ok"
    assert res.answer.startswith("华南")


def test_missing_api_key_returns_guidance(dataset, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert "DEEPSEEK_API_KEY" in run_agent("概况", dataset).answer
