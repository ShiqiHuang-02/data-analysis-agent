"""Upload handling, column metadata and a strictly limited filter interface."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
import math
import pandas as pd

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_ROWS = 250_000
MAX_COLUMNS = 150
ALLOWED_EXTENSIONS = {".csv", ".tsv", ".xlsx"}


@dataclass
class Dataset:
    df: pd.DataFrame
    filename: str = "dataset"

    def __post_init__(self) -> None:
        self.df = self.df.copy(deep=True)
        self._check_dimensions()

    def _check_dimensions(self) -> None:
        if self.df.shape[0] > MAX_ROWS or self.df.shape[1] > MAX_COLUMNS:
            raise ValueError(f"数据量超出限制：最多 {MAX_ROWS:,} 行、{MAX_COLUMNS} 列。")
        if self.df.columns.duplicated().any():
            raise ValueError("数据中有重复列名，请先修改列名。")
        self.df.columns = [str(c) for c in self.df.columns]

    @classmethod
    def from_bytes(cls, payload: bytes, filename: str) -> "Dataset":
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError("仅支持 CSV、TSV、XLSX 文件。")
        if not payload:
            raise ValueError("文件为空。")
        if len(payload) > MAX_UPLOAD_BYTES:
            raise ValueError("文件大小超过 15 MB 限制。")
        source = BytesIO(payload)
        try:
            if ext == ".xlsx":
                df = pd.read_excel(source, engine="openpyxl")
            else:
                delimiter = "\t" if ext == ".tsv" else ","
                try:
                    df = pd.read_csv(source, sep=delimiter, encoding="utf-8-sig", low_memory=False)
                except UnicodeDecodeError:
                    source.seek(0)
                    df = pd.read_csv(source, sep=delimiter, encoding="gb18030", low_memory=False)
        except Exception as exc:
            raise ValueError(f"文件读取失败：{type(exc).__name__}。请检查文件格式。") from exc
        if df.empty and len(df.columns) == 0:
            raise ValueError("文件没有可用数据。")
        return cls(df, filename)

    @property
    def column_names(self) -> list[str]:
        return list(self.df.columns)

    def ensure_column(self, name: str) -> None:
        if name not in self.df.columns:
            raise ValueError(f"列 {name!r} 不存在。可用列：{', '.join(self.column_names[:20])}")

    def ensure_numeric(self, name: str) -> None:
        self.ensure_column(name)
        if not pd.api.types.is_numeric_dtype(self.df[name]):
            raise ValueError(f"列 {name!r} 不是数值列。")

    def schema(self) -> dict[str, Any]:
        """No row samples or top values in model-visible schema."""
        return {
            "filename": self.filename,
            "row_count": int(len(self.df)),
            "column_count": int(self.df.shape[1]),
            "columns": [
                {"name": c, "dtype": str(self.df[c].dtype), "missing": int(self.df[c].isna().sum())}
                for c in self.df.columns
            ],
        }

    def filter_df(self, where: dict[str, Any] | None = None) -> pd.DataFrame:
        if not where:
            return self.df
        col = where["column"]
        op = where["operator"]
        val = where.get("value")
        self.ensure_column(col)
        s = self.df[col]
        if op == "is_null":
            mask = s.isna()
        elif op == "not_null":
            mask = s.notna()
        else:
            if val is None:
                raise ValueError(f"操作符 {op} 需要 value。")
            if pd.api.types.is_numeric_dtype(s):
                try:
                    v = float(val)
                except (ValueError, TypeError):
                    raise ValueError(f"数值列 {col!r} 的过滤值必须是数字。") from None
                if not math.isfinite(v):
                    raise ValueError("过滤值必须是有限数字。")
            elif pd.api.types.is_datetime64_any_dtype(s):
                v = pd.to_datetime(val, errors="raise")
            else:
                if op not in {"eq", "ne"}:
                    raise ValueError("字符串/分类列只支持 eq、ne、is_null、not_null。")
                v = str(val)
            ops = {
                "eq": lambda: s == v, "ne": lambda: s != v,
                "gt": lambda: s > v, "ge": lambda: s >= v,
                "lt": lambda: s < v, "le": lambda: s <= v,
            }
            mask = ops[op]()
        return self.df.loc[mask.fillna(False)]
