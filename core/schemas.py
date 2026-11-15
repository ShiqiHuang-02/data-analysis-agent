"""Pydantic inputs: reject unknown arguments and constrain tool choices."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FilterArgs(StrictArgs):
    column: str = Field(min_length=1, max_length=150)
    operator: Literal["eq", "ne", "gt", "ge", "lt", "le", "is_null", "not_null"] = "eq"
    value: str | float | int | None = None
    value_type: Literal["auto", "datetime"] = "auto"


class FilterGroupArgs(StrictArgs):
    logic: Literal["and", "or"] = "and"
    conditions: list[FilterArgs] = Field(min_length=1, max_length=8)


WhereArgs = FilterArgs | FilterGroupArgs


class OverviewArgs(StrictArgs):
    columns: list[str] | None = Field(default=None, max_length=20)


class MissingArgs(StrictArgs):
    pass


class ValueCountsArgs(StrictArgs):
    column: str = Field(min_length=1)
    top_n: int = Field(default=15, ge=1, le=30)
    where: WhereArgs | None = None


class GroupAggregateArgs(StrictArgs):
    group_by: str = Field(min_length=1)
    value_col: str | None = None
    agg: Literal["count", "sum", "mean", "median", "min", "max", "nunique"] = "count"
    where: WhereArgs | None = None
    top_n: int = Field(default=20, ge=1, le=30)
    ascending: bool = False

    @model_validator(mode="after")
    def check_value(self):
        if self.agg != "count" and not self.value_col:
            raise ValueError("agg 不为 count 时，value_col 必填。")
        return self


class CorrelationArgs(StrictArgs):
    columns: list[str] | None = Field(default=None, max_length=15)
    method: Literal["pearson", "spearman"] = "pearson"


class MetricArgs(StrictArgs):
    name: str = Field(min_length=1, max_length=80)
    agg: Literal["row_count", "count", "nunique", "sum", "mean", "median", "min", "max"]
    column: str | None = None

    @model_validator(mode="after")
    def check_column(self):
        if self.agg != "row_count" and not self.column:
            raise ValueError("除 row_count 外，每个指标必须指定 column。")
        if self.agg == "row_count" and self.column is not None:
            raise ValueError("row_count 统计行数，不接受 column。")
        return self


class AggregateTableArgs(StrictArgs):
    group_by: list[str] = Field(default_factory=list, max_length=3)
    metrics: list[MetricArgs] = Field(min_length=1, max_length=6)
    where: WhereArgs | None = None
    sort_by: str | None = None
    ascending: bool = False
    top_n: int = Field(default=20, ge=1, le=50)

    @model_validator(mode="after")
    def check_names(self):
        names = [metric.name for metric in self.metrics]
        if len(names) != len(set(names)) or len(self.group_by) != len(set(self.group_by)):
            raise ValueError("指标名称和分组字段不能重复。")
        if set(names) & set(self.group_by):
            raise ValueError("指标名称不能与分组字段相同。")
        if self.sort_by is not None and self.sort_by not in names:
            raise ValueError("sort_by 必须是一个指标名称。")
        return self


class ChartArgs(StrictArgs):
    kind: Literal["bar", "line", "hist", "scatter", "box"]
    x: str = Field(min_length=1)
    y: str | None = None
    agg: Literal["count", "sum", "mean", "median"] = "count"
    where: WhereArgs | None = None
    top_n: int = Field(default=15, ge=1, le=30)

    @model_validator(mode="after")
    def check_axes(self):
        if self.kind in {"scatter", "box", "line"} and not self.y:
            raise ValueError(f"{self.kind} 图需要 y 列。")
        if self.kind == "bar" and self.agg != "count" and not self.y:
            raise ValueError("非 count 柱状图需要 y 列。")
        return self


class TrendArgs(StrictArgs):
    date_col: str = Field(min_length=1, description="Date column, typically an ISO date string or datetime")
    value_col: str | None = Field(default=None, description="Numeric measure; required unless agg=count")
    frequency: Literal["day", "week", "month", "quarter"] = "month"
    agg: Literal["count", "sum", "mean", "median"] = "count"
    where: WhereArgs | None = None
    max_periods: int = Field(default=60, ge=1, le=120)

    @model_validator(mode="after")
    def check_metric(self):
        if self.agg != "count" and not self.value_col:
            raise ValueError("非 count 趋势聚合需要 value_col。")
        return self


class CleanPreviewArgs(StrictArgs):
    operation: Literal["drop_duplicates", "drop_missing", "fill_median", "fill_mode"]
    column: str | None = None

    @model_validator(mode="after")
    def check_column(self):
        if self.operation.startswith("fill_") and not self.column:
            raise ValueError("填充缺失值时需要 column。")
        return self


TOOL_SCHEMAS = {
    "overview": OverviewArgs,
    "missing_values": MissingArgs,
    "value_counts": ValueCountsArgs,
    "group_aggregate": GroupAggregateArgs,
    "correlation": CorrelationArgs,
    "make_chart": ChartArgs,
    "clean_preview": CleanPreviewArgs,
    "time_trend": TrendArgs,
    "aggregate_table": AggregateTableArgs,
}

TOOL_DESCRIPTIONS = {
    "overview": "Get dataset dimensions, numeric descriptive statistics and categorical column summaries. Never returns original rows.",
    "missing_values": "Count missing values for each column and their percentage.",
    "value_counts": "Count frequent values of a categorical column, with optional single or AND/OR filters. Use value_type=datetime for date-string comparisons.",
    "group_aggregate": "Aggregate records by a column (count means row count). Supports single or AND/OR filters. Use aggregate_table for multiple metrics or dimensions.",
    "correlation": "Calculate Pearson or Spearman correlations of numeric columns.",
    "make_chart": "Request a chart; chart rendering uses local data only. Supports bar/line/hist/scatter/box.",
    "clean_preview": "When user asks to clean/modify data, preview a specified cleaning operation and show changed row/missing counts. User must confirm in chat before the dataset changes.",
    "time_trend": "Time series descriptive analysis. Parse the date column, aggregate by day/week/month/quarter in chronological order, and return a line chart. Unlike basic make_chart(line), this understands dates. Does not forecast.",
    "aggregate_table": "Compute up to 6 named metrics grouped by 0-3 columns, with AND/OR filters. Empty group_by gives overall totals. row_count counts rows; count counts non-null values; nunique counts distinct non-null IDs; sum sums a numeric quantity (e.g. an existing order-count column). mean/median/min/max require numeric columns. All-missing sum is null. sort_by is a metric name; results may be truncated to top_n. Date-string filters require value_type=datetime. Do not invent business definitions; ask when ambiguous.",
}


def function_specs() -> list[dict]:
    return [
        {"type": "function", "function": {
            "name": name,
            "description": TOOL_DESCRIPTIONS[name],
            "parameters": schema.model_json_schema(),
        }}
        for name, schema in TOOL_SCHEMAS.items()
    ]
