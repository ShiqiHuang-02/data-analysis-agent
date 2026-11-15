"""Deterministic multi-dimensional aggregation with explicit metric definitions."""
from __future__ import annotations

import pandas as pd

from core.dataset import Dataset


METRIC_NOTES = {
    "row_count": "记录行数，包含缺失值；不自动等同于订单数或用户数。",
    "count": "指定字段的非缺失值个数。",
    "nunique": "指定字段的非缺失去重个数；各组去重数不可直接相加作为整体去重数。",
    "sum": "指定数值字段求和，跳过缺失；全缺失或空集返回空值。",
    "mean": "指定数值字段的非缺失记录均值，不自动代表客单价。",
    "median": "指定数值字段的非缺失记录中位数。",
    "min": "指定数值字段的非缺失最小值。",
    "max": "指定数值字段的非缺失最大值。",
}


def aggregate_table(ds: Dataset, args: dict) -> dict:
    dimensions = args["group_by"]
    for column in dimensions:
        ds.ensure_column(column)
    for metric in args["metrics"]:
        if metric["agg"] != "row_count":
            ds.ensure_column(metric["column"])
            if metric["agg"] not in {"count", "nunique"}:
                ds.ensure_numeric(metric["column"])

    df = ds.filter_df(args.get("where"))
    grouped = df.groupby(dimensions, dropna=False, observed=True, sort=False) if dimensions else None
    values = {}
    for metric in args["metrics"]:
        operation, column = metric["agg"], metric.get("column")
        if operation == "row_count":
            value = grouped.size() if grouped is not None else len(df)
        else:
            source = grouped[column] if grouped is not None else df[column]
            if operation == "sum":
                value = source.sum(min_count=1)
            elif operation == "nunique":
                value = source.nunique(dropna=True)
            else:
                value = getattr(source, operation)()
        values[metric["name"]] = value

    table = pd.DataFrame(values).reset_index() if dimensions else pd.DataFrame([values])
    total = len(table)
    if args.get("sort_by"):
        table = table.sort_values(args["sort_by"], ascending=args["ascending"],
                                  na_position="last", kind="stable")
    table = table.head(args["top_n"])
    # Convert pandas NA/NaT and missing group labels to JSON null.
    records = table.astype(object).where(table.notna(), None).to_dict("records")
    definitions = [{**metric, "definition": METRIC_NOTES[metric["agg"]],
                    "valid_rows": len(df) if metric["agg"] == "row_count"
                    else int(df[metric["column"]].notna().sum())}
                   for metric in args["metrics"]]
    return {"kind": "table", "table": records, "group_by": dimensions,
            "metric_definitions": definitions, "filtered_rows": len(df),
            "total_groups": total, "displayed_groups": len(records),
            "truncated": total > len(records),
            "note": "缺失分组值保留为空值；指标有效行数基于全部筛选后数据，非仅展示的 Top N。"}
