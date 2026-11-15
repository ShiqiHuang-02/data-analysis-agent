"""Exercise the offline user flow without sending any model requests."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest


def labeled(elements, label):
    return next(element for element in elements if element.label == label)


def test_advanced_ui_filters_and_data_revision():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app_streamlit.py"), default_timeout=30).run()
    assert not app.exception
    labeled(app.button, "运行多维汇总").click().run()
    assert not app.exception
    result = app.session_state["last_advanced"]["result"]
    assert result["status"] == "ok" and result["filtered_rows"] == 120
    assert result["total_groups"] == 12

    labeled(app.number_input, "条件数量").set_value(2).run()
    labeled(app.selectbox, "条件 1 字段").select("地区")
    labeled(app.text_input, "条件 1 值").set_value("华北")
    labeled(app.selectbox, "条件 2 字段").select("销售额")
    labeled(app.selectbox, "条件 2 比较").select("gt")
    labeled(app.text_input, "条件 2 值").set_value("100")
    labeled(app.button, "运行多维汇总").click().run()
    assert not app.exception
    result = app.session_state["last_advanced"]["result"]
    df = app.session_state["dataset"].df
    expected = df[(df["地区"] == "华北") & (df["销售额"] > 100)]
    assert result["filtered_rows"] == len(expected)
    assert sum(row["销售额_sum"] for row in result["table"]) == pytest.approx(expected["销售额"].sum())
    assert all(row["地区"] == "华北" for row in result["table"])
    old_revision = app.session_state["last_advanced"]["revision"]
    labeled(app.button, "恢复原始数据").click().run()
    assert not app.exception
    assert app.session_state["dataset_revision"] > old_revision
    assert not any("以下为上次运行" in caption.value for caption in app.caption)


def test_stats_display_survives_rerun_without_api():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app_streamlit.py"), default_timeout=30).run()
    app.session_state["messages"] = [{"role": "assistant", "content": "测试记录", "trace": [],
        "stats": {"elapsed_ms": 100, "model_requests": 1, "tools_executed": 0,
                  "total_tokens": None, "usage_status": "unavailable"}}]
    app.run()
    assert not app.exception
    assert any("Token 未返回" in caption.value for caption in app.caption)
    labeled(app.selectbox, "业务口径提示").select("general").run()
    assert not app.exception
    assert any("当前：通用分析" in caption.value for caption in app.caption)
