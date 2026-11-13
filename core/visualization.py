"""Local-only chart rendering from validated specs. Never executes generated code."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
import seaborn as sns

# Prefer an installed Chinese font; on systems without CJK fonts, plotting still works.
for _font in ("PingFang SC", "Hiragino Sans GB", "Noto Sans CJK SC", "Microsoft YaHei", "SimHei", "WenQuanYi Micro Hei"):
    try:
        font_manager.findfont(_font, fallback_to_default=False)
    except ValueError:
        continue
    plt.rcParams["font.sans-serif"] = [_font, "DejaVu Sans"]
    break
plt.rcParams["axes.unicode_minus"] = False
from core.dataset import Dataset
from core.schemas import ChartArgs, TrendArgs
from core.trends import summarize_trend


def build_chart(ds: Dataset, spec: dict):
    if spec.get("kind") == "trend":
        return _build_time_trend(ds, spec)
    a = ChartArgs.model_validate(spec).model_dump(exclude_none=True)
    df = ds.filter_df(a.get("where"))
    x, y, kind = a["x"], a.get("y"), a["kind"]
    ds.ensure_column(x)
    if y:
        ds.ensure_column(y)
    if df.empty:
        raise ValueError("无法绘制空数据集。")
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    try:
        if kind in {"bar", "line"}:
            if a["agg"] == "count":
                data = df.groupby(x, dropna=False).size().rename("count")
                ylabel = "Count"
            else:
                data = df.groupby(x, dropna=False)[y].agg(a["agg"])
                ylabel = f"{y} ({a['agg']})"
            if kind == "bar":
                data = data.sort_values(ascending=False).head(a["top_n"])
                ax.bar(data.index.astype(str), data.values)
                ax.tick_params(axis="x", labelrotation=30)
            else:
                # Sort datetime x values chronologically when possible; otherwise lexical.
                data = data.sort_index().head(a["top_n"])
                ax.plot(data.index.astype(str), data.values, marker="o")
                ax.tick_params(axis="x", labelrotation=30)
            ax.set_ylabel(ylabel)
        elif kind == "hist":
            sns.histplot(data=df, x=x, bins=20, ax=ax)
            ax.set_ylabel("Count")
        elif kind == "scatter":
            sns.scatterplot(data=df.sample(min(len(df), 2000), random_state=42), x=x, y=y, ax=ax, s=25)
        elif kind == "box":
            # If x is numeric/high cardinality, reject to prevent unreadable plots.
            if df[x].nunique() > 30:
                raise ValueError("箱线图 x 类别超过 30，建议更换分组列。")
            sns.boxplot(data=df, x=x, y=y, ax=ax)
            ax.tick_params(axis="x", labelrotation=30)
        ax.set_title(f"{kind.capitalize()} · {x}" + (f" / {y}" if y else ""))
        ax.set_xlabel(x)
        return fig
    except Exception:
        plt.close(fig)
        raise


def _build_time_trend(ds: Dataset, spec: dict):
    # Recompute through the same validated path as the tool: no blind use of model values.
    a = TrendArgs.model_validate({k: v for k, v in spec.items() if k != "kind"}).model_dump(exclude_none=True)
    data = summarize_trend(ds, a)
    rows = data["table"]
    if not rows:
        raise ValueError("没有可绘制的时间分组。")
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    try:
        times = pd.to_datetime([r["period"] for r in rows])
        values = [r[data["metric"]] for r in rows]
        ax.plot(times, values, marker="o")
        ax.set_xlabel(f"{a['date_col']} ({a['frequency']})")
        ax.set_ylabel(data["metric"])
        ax.set_title(f"Time Trend · {data['metric']}")
        fig.autofmt_xdate()
        return fig
    except Exception:
        plt.close(fig)
        raise
