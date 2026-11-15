"""Read-only SQL tool: execution, SELECT-only enforcement, row cap."""
import pytest

from core.dataset import Dataset
from core.sql import sql_query
from core.tools import run_tool


@pytest.fixture
def ds():
    import pandas as pd
    return Dataset(pd.DataFrame({
        "地区": ["华北", "华北", "华南", "华南", "华东"],
        "渠道": ["官网", "门店", "官网", "门店", "电商"],
        "销售额": [100.0, 200.0, 300.0, 400.0, None],
    }), "sales.csv")


def test_select_group_and_sum(ds):
    r = sql_query(ds, {"query": 'SELECT 地区, SUM(销售额) AS s FROM data GROUP BY 地区 ORDER BY s DESC'})
    assert r["kind"] == "table"
    assert r["table"][0] == {"地区": "华南", "s": 700.0}


def test_select_star_chinese_columns(ds):
    r = sql_query(ds, {"query": 'SELECT * FROM data ORDER BY 销售额 DESC LIMIT 1'})
    assert r["table"][0]["地区"] == "华南"


def test_row_cap(ds):
    r = sql_query(ds, {"query": "SELECT * FROM data", "max_rows": 2})
    assert r["truncated"] is True
    assert len(r["table"]) == 2


def test_non_select_rejected(ds):
    for q in ["INSERT INTO data VALUES ('x')", "UPDATE data SET 销售额=0",
              "DELETE FROM data", "DROP TABLE data", "CREATE TABLE x(a)"]:
        with pytest.raises(ValueError):
            sql_query(ds, {"query": q})


def test_multi_statement_rejected(ds):
    with pytest.raises(ValueError):
        sql_query(ds, {"query": "SELECT 1; SELECT 2"})


def test_bad_sql_raises_valueerror(ds):
    with pytest.raises(ValueError):
        sql_query(ds, {"query": "SELECT 不存在列 FROM data"})


def test_tool_registration(ds):
    r = run_tool(ds, "sql_query", {"query": "SELECT COUNT(*) AS n FROM data"})
    assert r["status"] == "ok"
    assert r["table"] == [{"n": 5}]
