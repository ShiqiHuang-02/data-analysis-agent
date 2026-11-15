"""Transparent, rule-based business hints; no training or inferred metric mapping."""
from __future__ import annotations

from core.dataset import Dataset


PROFILE_LABELS = {"auto": "自动识别", "general": "通用分析", "sales": "销售分析"}
SALES_HINT = """销售分析口径：
- 先判断一行是订单、订单明细还是汇总记录；字段名只是线索，不能确定粒度时先询问。
- 明确的订单数字段（如订单数）通常需要 sum；订单编号统计用 nunique；记录行数用 row_count。不得把三者混用。
- 销售额、实付金额、退款金额、GMV 不自动等价，未提供口径时不可自行扣除退款或推断 GMV。
- 客单价需先明确分子金额和分母订单口径，不得直接用每行金额均值代替。
- 同时要求多个维度/指标时使用 aggregate_table，逐一命名指标并说明缺失值处理。
- 只展示 Top N 时说明截断；组内去重计数之和不等于整体去重计数。
- 相关关系、地区差异和时间变化只能描述，不能据此认定增长原因或营销因果效果。"""


def resolve_profile(ds: Dataset, mode: str = "auto") -> dict[str, str]:
    if mode not in PROFILE_LABELS:
        raise ValueError("未知分析场景。")
    names = {column.strip().casefold() for column in ds.column_names}
    amount = names & {"销售额", "销售金额", "实付金额", "sales", "revenue", "sales_amount"}
    order = names & {"订单数", "订单号", "订单编号", "order_id", "order_count"}
    selected = "sales" if mode == "sales" or (mode == "auto" and amount and order) else "general"
    reason = ("字段同时包含销售金额和订单线索；仅加载口径提示，仍需核对数据粒度。"
              if mode == "auto" and selected == "sales" else
              "未命中销售字段组合，使用通用分析。" if mode == "auto" else "由用户手动选择。")
    return {"id": selected, "label": PROFILE_LABELS[selected], "reason": reason,
            "instructions": SALES_HINT if selected == "sales" else "通用分析：指标含义或数据粒度不明确时，先询问，不自行设定业务口径。"}
