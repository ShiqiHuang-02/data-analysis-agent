"""User-confirmed cleaning proposals; a model response cannot mutate a Dataset."""
from __future__ import annotations

from typing import Any
from core.dataset import Dataset
from core.schemas import CleanPreviewArgs
from core.tools import compute_clean, run_tool


def proposal_from_trace(trace: list[dict[str, Any]], revision: int) -> dict | None:
    """Pick the last *successful* model-requested cleaning preview."""
    for item in reversed(trace):
        result = item.get("result", {})
        if item.get("tool") == "clean_preview" and result.get("status") == "ok":
            return {"revision": revision, "arguments": item["arguments"], "preview": result}
    return None


def confirm_proposal(dataset: Dataset, proposal: dict, revision: int) -> Dataset:
    """Fail closed if another upload, clean, or reset invalidated the preview."""
    if proposal.get("revision") != revision:
        raise ValueError("数据已经发生变化，请重新生成清洗预览。")
    args = CleanPreviewArgs.model_validate(proposal["arguments"]).model_dump(exclude_none=True)
    preview = run_tool(dataset, "clean_preview", args)
    if preview.get("status") != "ok":
        raise ValueError(preview.get("error", "清洗预览失效。"))
    # Revalidate to guard against modified/stale content before applying changes.
    for field in ("before_rows", "after_rows", "before_missing", "after_missing"):
        if preview[field] != proposal["preview"].get(field):
            raise ValueError("当前数据与预览时不同，请重新生成预览。")
    return Dataset(compute_clean(dataset, args), dataset.filename)
