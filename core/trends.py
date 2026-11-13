"""Time-aware aggregation shared by the tool and the chart renderer.

No predictions or interpolation are inferred from the input data. Missing
calendar periods are not filled with invented zeroes.
"""
from __future__ import annotations

import pandas as pd
from core.dataset import Dataset

FREQUENCIES = {"day": "D", "week": "W-SUN", "month": "M", "quarter": "Q"}


def summarize_trend(ds: Dataset, args: dict) -> dict:
    date_col = args["date_col"]
    ds.ensure_column(date_col)
    agg = args["agg"]
    value_col = args.get("value_col")
    if agg != "count":
        if not value_col:
            raise ValueError("统计求和/均值/中位数时必须指定 value_col。")
        ds.ensure_numeric(value_col)
    df = ds.filter_df(args.get("where"))
    if df.empty:
        raise ValueError("筛选后没有数据，无法计算时间趋势。")
    if pd.api.types.is_numeric_dtype(df[date_col]):
        raise ValueError("日期列是数值类型，请先将其转为日期字符串（如 2026-11-01）。")
    # Mixed format lets common ISO, slash and Excel-imported datetime values coexist.
    try:
        dates = pd.to_datetime(df[date_col], errors="coerce", format="mixed")
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("日期列无法解析。请使用统一的日期格式。") from exc
    if isinstance(dates.dtype, pd.DatetimeTZDtype):
        dates = dates.dt.tz_localize(None)
    valid = dates.notna()
    if not valid.any():
        raise ValueError("日期列没有可解析的日期。")
    periods = dates.loc[valid].dt.to_period(FREQUENCIES[args["frequency"]]).dt.start_time
    valid_df = df.loc[valid]
    if agg == "count":
        values = valid_df.groupby(periods, sort=True).size()
        metric = "count"
    else:
        grouped = valid_df.groupby(periods, sort=True)[value_col]
        if agg == "sum":
            values = grouped.sum(min_count=1)
        elif agg == "mean":
            values = grouped.mean()
        elif agg == "median":
            values = grouped.median()
        else:
            raise ValueError("不支持的趋势聚合方式。")
        metric = f"{value_col}_{agg}"
    total = len(values)
    max_periods = args["max_periods"]
    # Always display the most recent time periods, not a lexically sorted Top N.
    if total > max_periods:
        values = values.tail(max_periods)
    records = [{"period": p.strftime("%Y-%m-%d"), metric: None if pd.isna(v) else float(v) if agg != "count" else int(v)}
               for p, v in values.items()]
    return {
        "kind": "chart", "table": records,
        "chart_spec": {"kind": "trend", **args},
        "date_col": date_col, "frequency": args["frequency"], "agg": agg,
        "metric": metric, "filtered_rows": len(df),
        "valid_date_rows": int(valid.sum()), "invalid_date_rows": int((~valid).sum()),
        "total_periods": total, "displayed_periods": len(records),
        "truncated": total > max_periods,
        "note": "只展示实际出现的日期分组；未出现的日期不自动补 0。",
    }
