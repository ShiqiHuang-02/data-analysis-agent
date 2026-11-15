"""Read-only SQL over an in-memory SQLite copy of the uploaded dataset.

This is the "Text-to-SQL" branch of the agent: the model writes SQL instead of
picking a fixed tool. Safety model (defense in depth):

- The SQLite connection is ``:memory:`` (nothing touches disk).
- The table is created first, then ``PRAGMA query_only=ON`` blocks any DML/DDL.
- Only a single leading ``SELECT`` is accepted; multi-statement is rejected.
- Results are capped and a VM-instruction budget aborts pathological queries.

See docs/DESIGN.md for the trade-off against the fixed tool allowlist.
"""
from __future__ import annotations

import re
import sqlite3

from core.dataset import Dataset

MAX_ROWS = 200
MAX_QUERY_CHARS = 4000
TABLE_NAME = "data"

_SELECT_RE = re.compile(r"^\s*(?:--[^\n]*\n|/\*.*?\*/|\s)*select\b", re.IGNORECASE | re.DOTALL)


def _too_complex() -> None:
    raise sqlite3.OperationalError("SQL 过于复杂或执行过慢，已中止。")


def _ensure_single_select(query: str) -> str:
    if len(query) > MAX_QUERY_CHARS:
        raise ValueError(f"SQL 长度超过 {MAX_QUERY_CHARS} 字符限制。")
    if not _SELECT_RE.match(query):
        raise ValueError("只允许单条 SELECT 查询。")
    # 去掉末尾分号后若仍含分号，则视为多语句，拒绝。
    stripped = query.strip().rstrip(";").strip()
    if not stripped:
        raise ValueError("SQL 查询为空。")
    if ";" in stripped:
        raise ValueError("只允许单条 SELECT，不能包含多个语句。")
    return stripped


def sql_query(ds: Dataset, a: dict) -> dict:
    """Execute one read-only SELECT against the in-memory copy of ``ds.df``."""
    query = a["query"]
    max_rows = a.get("max_rows", 50)
    q = _ensure_single_select(query)
    conn = sqlite3.connect(":memory:")
    try:
        ds.df.to_sql(TABLE_NAME, conn, index=False, if_exists="replace")
        conn.execute("PRAGMA query_only = ON")
        conn.set_progress_handler(_too_complex, 200_000)
        try:
            cur = conn.execute(q)
        except sqlite3.Error as exc:
            # 只回传精简报错，供模型自行修正；不含路径或原始数据。
            raise ValueError(f"SQL 执行失败：{str(exc).strip()[:300]}") from exc
        cols = [d[0] for d in cur.description]
        rows = cur.fetchmany(max_rows + 1)
        truncated = len(rows) > max_rows
        return {
            "kind": "table",
            "table": [dict(zip(cols, r)) for r in rows[:max_rows]],
            "columns": cols,
            "row_count": len(rows),
            "truncated": truncated,
            "note": f"SQL 只读执行，表名固定为 {TABLE_NAME}，最多返回 {max_rows} 行；"
                    + ("结果已截断。" if truncated else ""),
        }
    finally:
        conn.close()
