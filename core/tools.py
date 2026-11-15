"""Explicit read-only pandas tools; no eval/exec, file access or user Python."""
from __future__ import annotations

from typing import Any
import math
import pandas as pd
from pydantic import ValidationError

from core.dataset import Dataset
from core.schemas import TOOL_SCHEMAS
from core.trends import summarize_trend
from core.aggregation import aggregate_table


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, (pd.Timestamp, pd.Period)):
        return str(value)
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        return _safe(value.item())
    return str(value)


def _where(args: dict) -> dict | None:
    return args["where"] if args.get("where") else None


def overview(ds: Dataset, a: dict) -> dict:
    df = ds.df
    columns = a.get("columns") or ds.column_names[:30]
    data: list[dict] = []
    for col in columns:
        ds.ensure_column(col)
        s = df[col]
        stat: dict[str, Any] = {"column": col, "dtype": str(s.dtype), "missing": int(s.isna().sum()), "unique": int(s.nunique(dropna=True))}
        if pd.api.types.is_numeric_dtype(s):
            stat.update({"mean": _safe(float(s.mean())), "median": _safe(float(s.median())), "min": _safe(s.min()), "max": _safe(s.max())})
        data.append(stat)
    return {"kind": "overview", "row_count": len(df), "column_count": df.shape[1], "table": data,
            "note": "仅返回统计摘要，无原始数据行。"}


def missing_values(ds: Dataset, a: dict) -> dict:
    n = len(ds.df)
    data = [{"column": c, "missing": int(ds.df[c].isna().sum()),
             "pct": round(100 * ds.df[c].isna().sum() / n, 2) if n else 0.0}
            for c in ds.column_names]
    return {"kind": "table", "table": data[:100], "row_count": n}


def value_counts(ds: Dataset, a: dict) -> dict:
    col = a["column"]
    ds.ensure_column(col)
    s = ds.filter_df(_where(a))[col]
    # Deliberately no row-level samples, only value counts (potentially sensitive labels).
    vc = s.value_counts(dropna=False).head(a["top_n"])
    data = [{"value": str(k), "count": int(v)} for k, v in vc.items()]
    return {"kind": "table", "table": data, "column": col, "filtered_rows": len(s)}


def group_aggregate(ds: Dataset, a: dict) -> dict:
    by = a["group_by"]
    ds.ensure_column(by)
    val = a.get("value_col")
    agg = a["agg"]
    df = ds.filter_df(_where(a))
    if agg == "count":
        result = df.groupby(by, dropna=False).size()
        metric = "count"
    else:
        ds.ensure_column(val)
        if agg in {"sum", "mean", "median"}:
            ds.ensure_numeric(val)
        result = df.groupby(by, dropna=False)[val].agg(agg)
        metric = f"{val}_{agg}"
    # Sorted by metric to produce immediately meaningful Top N.
    result = result.sort_values(ascending=a["ascending"], na_position="last").head(a["top_n"])
    records = [{"group": str(k), metric: _safe(v)} for k, v in result.items()]
    return {"kind": "table", "table": records, "metric": metric,
            "filtered_rows": len(df), "group_by": by, "agg": agg}


def correlation(ds: Dataset, a: dict) -> dict:
    cols = a.get("columns") or list(ds.df.select_dtypes(include="number").columns)[:15]
    if not cols:
        raise ValueError("没有可用数值列。")
    for c in cols:
        ds.ensure_numeric(c)
    matrix = ds.df[cols].corr(method=a["method"])
    records = [{"column": c, **{x: _safe(float(matrix.loc[c, x])) for x in cols}} for c in cols]
    return {"kind": "table", "table": records, "method": a["method"]}


def make_chart(ds: Dataset, a: dict) -> dict:
    ds.ensure_column(a["x"])
    if a.get("y"):
        ds.ensure_column(a["y"])
    kind = a["kind"]
    if kind in {"scatter", "hist"}:
        ds.ensure_numeric(a["x"])
    if kind in {"scatter", "box"}:
        ds.ensure_numeric(a["y"])
    if kind in {"bar", "line"} and a["agg"] != "count":
        ds.ensure_numeric(a["y"])
    df = ds.filter_df(_where(a))
    if df.empty:
        raise ValueError("筛选后没有数据，无法作图。")
    return {"kind": "chart", "chart_spec": a, "filtered_rows": len(df),
            "note": "图表由本地代码绘制，不向 LLM 发送用于绘图的原始行。"}


def compute_clean(ds: Dataset, a: dict) -> pd.DataFrame:
    df = ds.df.copy(deep=True)
    op, col = a["operation"], a.get("column")
    if col:
        ds.ensure_column(col)
    if op == "drop_duplicates":
        return df.drop_duplicates()
    if op == "drop_missing":
        return df.dropna(subset=[col] if col else None)
    if op == "fill_median":
        ds.ensure_numeric(col)
        med = df[col].median()
        if pd.isna(med):
            raise ValueError("该列全部为缺失值，不能使用中位数填充。")
        df[col] = df[col].fillna(med)
        return df
    if op == "fill_mode":
        mode = df[col].mode(dropna=True)
        if mode.empty:
            raise ValueError("该列没有非缺失值，不能使用众数填充。")
        df[col] = df[col].fillna(mode.iloc[0])
        return df
    raise ValueError("未知清洗操作。")


def clean_preview(ds: Dataset, a: dict) -> dict:
    after = compute_clean(ds, a)
    return {"kind": "clean_preview", "operation": a["operation"], "column": a.get("column"),
            "before_rows": len(ds.df), "after_rows": len(after),
            "before_missing": int(ds.df.isna().sum().sum()), "after_missing": int(after.isna().sum().sum()),
            "note": "只是预览，没有修改数据。必须在界面上点击「应用清洗」才会生效。"}


FUNCTIONS = {
    "overview": overview,
    "missing_values": missing_values,
    "value_counts": value_counts,
    "group_aggregate": group_aggregate,
    "correlation": correlation,
    "make_chart": make_chart,
    "clean_preview": clean_preview,
    "time_trend": summarize_trend,
    "aggregate_table": aggregate_table,
}


def run_tool(ds: Dataset, name: str, arguments: dict) -> dict:
    """Fail closed: exact allowlist + Pydantic validation, normalized JSON result."""
    if name not in FUNCTIONS:
        return {"status": "error", "error": f"不允许调用工具 {name!r}。"}
    try:
        parsed = TOOL_SCHEMAS[name].model_validate(arguments)
        payload = FUNCTIONS[name](ds, parsed.model_dump(exclude_none=True))
        if hasattr(parsed, "where") and parsed.where is not None:
            payload["applied_filter"] = parsed.where.model_dump(exclude_none=True)
        return {"status": "ok", **_safe(payload)}
    except (ValidationError, ValueError, KeyError, TypeError, OverflowError) as exc:
        return {"status": "error", "error": str(exc)[:500],
                "hint": "请核对列名、数据类型和工具参数，再尝试其他允许的调用。"}
    except Exception:
        # Do not send local tracebacks, paths or arbitrary upload data to remote service.
        return {"status": "error", "error": "数据操作执行失败。请检查列类型及筛选条件。"}
