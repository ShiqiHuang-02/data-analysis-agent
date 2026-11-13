"""Streamlit local-first data analysis UI."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import hashlib
import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from core.dataset import Dataset
from core.agent import run_agent
from core.cleaning import proposal_from_trace, confirm_proposal
from core.schemas import TOOL_SCHEMAS
from core.tools import compute_clean, run_tool
from core.visualization import build_chart

load_dotenv()
st.set_page_config(page_title="Data Analysis Agent", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
.block-container {padding-top: 2rem; max-width: 1220px}
[data-testid="stMetricValue"] {font-size: 1.45rem}
.hero {padding: 20px 26px; border: 1px solid #cbd5e1; border-radius: 16px; margin-bottom: 18px}
.hero h1 {margin: 0; letter-spacing: -.04em}
.hero p {margin: 8px 0 0; opacity: .76}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>📊 Data Analysis Agent</h1><p>DeepSeek × Function Calling × Pandas · 可解释、可复核的自然语言数据分析</p></div>', unsafe_allow_html=True)

SAMPLE_PATH = Path(__file__).parent / "sample_data" / "sales_demo.csv"


def load_data():
    with st.sidebar:
        st.subheader("📁 数据源")
        upload = st.file_uploader("上传 CSV / TSV / Excel", type=["csv", "tsv", "xlsx"])
        st.caption("最大 15 MB / 250,000 行 / 150 列")
        if upload:
            payload = upload.getvalue()
            filename = upload.name
        else:
            payload = SAMPLE_PATH.read_bytes()
            filename = "sales_demo.csv"
            st.info("当前使用内置销售示例数据，可直接离线体验。")
        key = hashlib.sha256(payload + filename.encode()).hexdigest()
        if st.session_state.get("dataset_key") != key:
            try:
                dataset = Dataset.from_bytes(payload, filename)
            except ValueError as exc:
                st.error(str(exc))
                st.stop()
            st.session_state["dataset"] = dataset
            st.session_state["original_dataset"] = Dataset(dataset.df, dataset.filename)
            st.session_state["dataset_key"] = key
            st.session_state["messages"] = []
            st.session_state["dataset_revision"] = 0
            st.session_state["pending_clean"] = None
            st.session_state["last_offline"] = None
        st.divider()
        st.subheader("🤖 模型配置")
        configured = bool(os.getenv("DEEPSEEK_API_KEY", "").strip())
        st.success("API Key 已配置") if configured else st.warning("未配置 API Key：聊天功能不可用，离线工具仍可用")
        model = st.text_input("模型名称", value=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"))
        st.caption("不要在聊天框或代码仓库中粘贴真实 API Key。")
        st.divider()
        if st.button("清空当前对话", use_container_width=True):
            st.session_state["messages"] = []
            st.session_state["pending_clean"] = None
            st.rerun()
        if st.button("恢复原始数据", use_container_width=True):
            original = st.session_state["original_dataset"]
            st.session_state["dataset"] = Dataset(original.df, original.filename)
            st.session_state["messages"] = []
            st.session_state["dataset_revision"] += 1
            st.session_state["pending_clean"] = None
            st.session_state["last_offline"] = None
            st.rerun()
    return st.session_state["dataset"], model


def render_result(result: dict, *, key: str):
    if result.get("status") != "ok":
        st.error(result.get("error", "操作未成功"))
        return
    kind = result.get("kind")
    if "table" in result and result["table"]:
        table = pd.DataFrame(result["table"])
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.download_button("下载此结果 CSV", data=table.to_csv(index=False).encode("utf-8-sig"),
                           file_name="analysis_result.csv", mime="text/csv", key=f"dl_{key}")
    if kind == "chart":
        try:
            fig = build_chart(st.session_state["dataset"], result["chart_spec"])
            st.pyplot(fig, use_container_width=True)
            png = BytesIO()
            fig.savefig(png, format="png", dpi=160, bbox_inches="tight")
            st.download_button("下载图表 PNG", png.getvalue(), file_name="analysis_chart.png",
                               mime="image/png", key=f"png_{key}")
            plt.close(fig)
        except Exception as exc:
            st.error(f"绘图失败：{exc}")
    elif kind == "clean_preview":
        st.info(f"清洗预览：行数 {result['before_rows']} → {result['after_rows']}；缺失值 {result['before_missing']} → {result['after_missing']}。如果由 Agent 发起，请在「Agent 对话」下方确认应用。")
    if result.get("chart_spec", {}).get("kind") == "trend":
        st.caption(f"时间趋势：有效日期 {result['valid_date_rows']} 行、无效/缺失日期 {result['invalid_date_rows']} 行；共 {result['total_periods']} 个周期。未出现的日期不补零。" +
                   ("仅展示最近的部分周期。" if result.get("truncated") else ""))
    if kind in {"overview", "table"}:
        st.caption(f"统计基于本地 Pandas 执行；当前数据行数：{len(st.session_state['dataset'].df):,}")


ds, selected_model = load_data()
cols = st.columns(4)
with cols[0]: st.metric("数据行数", f"{len(ds.df):,}")
with cols[1]: st.metric("字段数量", ds.df.shape[1])
with cols[2]: st.metric("缺失单元格", f"{int(ds.df.isna().sum().sum()):,}")
with cols[3]: st.metric("重复记录", f"{int(ds.df.duplicated().sum()):,}")

st.caption("🔐 本地读取原始文件，发送给 DeepSeek 的内容包括你的问题、列名与类型、聚合统计结果；请勿上传或查询未经授权的敏感数据。")
chat_tab, tools_tab, cleaning_tab, raw_tab = st.tabs(["💬 Agent 对话", "📈 离线分析", "🧹 数据清洗", "🗃️ 数据预览"])

with chat_tab:
    st.caption("试试：‘查看数据概况’、‘按地区统计销售额’、‘按月统计销售额趋势’、‘用中位数填充评分列缺失值’")
    for i, item in enumerate(st.session_state["messages"]):
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            if item["role"] == "assistant" and item.get("trace"):
                with st.expander(f"🔎 工具调用记录（{len(item['trace'])} 次）", expanded=False):
                    for j, entry in enumerate(item["trace"]):
                        st.code(f"{entry['tool']}({entry['arguments']})", language="python")
                        render_result(entry["result"], key=f"history_{i}_{j}")
    question = st.chat_input("用自然语言问这份数据...")
    if question:
        previous = [{"role": m["role"], "content": m["content"]} for m in st.session_state["messages"]]
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("正在解析问题并调用本地分析工具..."):
                result = run_agent(question, ds, history=previous, model=selected_model)
            st.markdown(result.answer)
            if result.trace:
                with st.expander(f"🔎 本轮工具调用记录（{len(result.trace)} 次）", expanded=True):
                    for j, entry in enumerate(result.trace):
                        st.code(f"{entry['tool']}({entry['arguments']})", language="python")
                        render_result(entry["result"], key=f"new_{len(previous)}_{j}")
        st.session_state["messages"].append({"role": "assistant", "content": result.answer, "trace": result.trace})
        proposal = proposal_from_trace(result.trace, st.session_state["dataset_revision"])
        if proposal:
            st.session_state["pending_clean"] = proposal

    pending = st.session_state.get("pending_clean")
    if pending is not None:
        st.divider()
        st.subheader("🧹 Agent 提议的清洗操作 — 等待你确认")
        detail = pending["preview"]
        st.info(f"操作：{detail['operation']}" + (f"，字段：{detail['column']}" if detail.get("column") else "") +
                f"。行数 {detail['before_rows']} → {detail['after_rows']}；缺失单元格 {detail['before_missing']} → {detail['after_missing']}。")
        approve_col, cancel_col = st.columns(2)
        with approve_col:
            if st.button("✅ 确认应用这次清洗", type="primary", use_container_width=True, key="confirm_agent_clean"):
                try:
                    updated = confirm_proposal(st.session_state["dataset"], pending, st.session_state["dataset_revision"])
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.session_state["dataset"] = updated
                    st.session_state["dataset_revision"] += 1
                    st.session_state["pending_clean"] = None
                    st.session_state["messages"] = []  # prevent showing stale analysis for updated data
                    st.session_state["last_offline"] = None
                    st.rerun()
        with cancel_col:
            if st.button("取消，不修改数据", use_container_width=True, key="cancel_agent_clean"):
                st.session_state["pending_clean"] = None
                st.rerun()

with tools_tab:
    st.subheader("无需 API Key 的分析工具")
    st.caption("这些操作直接执行 Pandas，既可体验系统，也能独立核对 Agent 的分析结果。")
    colA, colB = st.columns(2)
    with colA:
        task = st.selectbox("分析任务", ["数据概况", "缺失值统计", "取值频数", "分组聚合", "数值相关性", "时间趋势分析", "绘制图表"])
        target = st.selectbox("字段", ds.column_names)
    with colB:
        numeric = list(ds.df.select_dtypes(include="number").columns)
        value_col = st.selectbox("数值字段", numeric or ["（没有数值列）"])
        agg = st.selectbox("聚合方式", ["count", "sum", "mean", "median", "min", "max", "nunique"])
        kind = st.selectbox("图表类型", ["bar", "line", "hist", "scatter", "box"])
    if task == "时间趋势分析":
        time_left, time_right = st.columns(2)
        with time_left:
            date_candidates = [col for col in ds.column_names if "日期" in col or "时间" in col or "date" in col.lower()]
            date_col = st.selectbox("日期字段", date_candidates + [c for c in ds.column_names if c not in date_candidates])
        with time_right:
            freq_label = st.selectbox("时间粒度", ["按日", "按周", "按月", "按季度"], index=2)
            trend_agg = st.selectbox("趋势聚合方式", ["count", "sum", "mean", "median"], index=1)
        freq = {"按日": "day", "按周": "week", "按月": "month", "按季度": "quarter"}[freq_label]
        st.caption("时间聚合会解析日期、按时间排序；未出现的时间段不自动填零。")
    if st.button("运行分析", type="primary"):
        if task == "数据概况": name, args = "overview", {}
        elif task == "缺失值统计": name, args = "missing_values", {}
        elif task == "取值频数": name, args = "value_counts", {"column": target}
        elif task == "分组聚合":
            name = "group_aggregate"
            args = {"group_by": target, "value_col": value_col if numeric else None, "agg": agg}
        elif task == "数值相关性": name, args = "correlation", {}
        elif task == "时间趋势分析":
            name = "time_trend"
            args = {"date_col": date_col, "frequency": freq, "agg": trend_agg,
                    **({"value_col": value_col} if trend_agg != "count" and numeric else {})}
        else:
            name = "make_chart"
            x = value_col if kind == "hist" and numeric else target
            y = value_col if kind in {"scatter", "line", "box"} or (kind == "bar" and agg != "count") else None
            args = {"kind": kind, "x": x, "y": y, "agg": agg if kind in {"bar", "line"} else "count"}
        response = run_tool(ds, name, args)
        st.session_state["last_offline"] = {"name": name, "args": args, "result": response,
                                            "revision": st.session_state["dataset_revision"]}
    last_offline = st.session_state.get("last_offline")
    if last_offline and last_offline["revision"] == st.session_state["dataset_revision"]:
        st.caption(f"当前工具：{last_offline['name']}")
        render_result(last_offline["result"], key="offline_run")

with cleaning_tab:
    st.subheader("可预览、可确认的数据清洗")
    st.caption("LLM 没有权限直接修改 DataFrame。这里由你明确确认后才生效。")
    operation_map = {"删除重复行": "drop_duplicates", "删除缺失行": "drop_missing", "数值缺失填充中位数": "fill_median", "缺失填充众数": "fill_mode"}
    operation_label = st.selectbox("清洗操作", list(operation_map))
    operation = operation_map[operation_label]
    use_col = operation in {"fill_median", "fill_mode", "drop_missing"}
    if operation == "fill_median": options = list(ds.df.select_dtypes(include="number").columns)
    else: options = ds.column_names
    if use_col and not options:
        st.warning("当前没有符合条件的列。")
        column = None
    else:
        column = st.selectbox("作用列", options) if use_col else None
    args = {"operation": operation, **({"column": column} if column else {})}
    preview = run_tool(ds, "clean_preview", args)
    if preview.get("status") == "ok":
        st.info(f"执行前：{preview['before_rows']:,} 行 / {preview['before_missing']:,} 个缺失值；执行后：{preview['after_rows']:,} 行 / {preview['after_missing']:,} 个缺失值。")
    else:
        st.warning(preview.get("error", "无法预览"))
    if st.button("✅ 确认应用清洗", disabled=preview.get("status") != "ok", type="primary"):
        cleaned = compute_clean(ds, args)
        st.session_state["dataset"] = Dataset(cleaned, ds.filename)
        st.session_state["messages"] = []
        st.session_state["dataset_revision"] += 1
        st.session_state["pending_clean"] = None
        st.session_state["last_offline"] = None
        st.rerun()
    st.download_button("下载当前清洗后的 CSV", data=ds.df.to_csv(index=False).encode("utf-8-sig"),
                       file_name="cleaned_dataset.csv", mime="text/csv")

with raw_tab:
    st.subheader(f"{ds.filename}")
    st.dataframe(ds.df.head(200), use_container_width=True, hide_index=True)
    st.caption("仅在本地展示前 200 行；不会因为打开此页而发送给模型。")
    st.subheader("Schema")
    st.dataframe(pd.DataFrame(ds.schema()["columns"]), use_container_width=True, hide_index=True)
