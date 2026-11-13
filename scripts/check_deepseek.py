"""Optional paid live test: verifies an actual model -> tool -> model round trip.

Run only after setting DEEPSEEK_API_KEY in .env. This makes real API requests.
"""
from __future__ import annotations

from pathlib import Path
import os
import sys

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.agent import run_agent
from core.dataset import Dataset


def main() -> int:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    if not os.getenv("DEEPSEEK_API_KEY", "").strip():
        print("尚未配置 DEEPSEEK_API_KEY：未发起网络请求。请填写 .env 后重新运行。")
        return 2
    project = Path(__file__).resolve().parents[1]
    ds = Dataset.from_bytes((project / "sample_data/sales_demo.csv").read_bytes(), "sales_demo.csv")
    result = run_agent(
        "请使用 group_aggregate 工具计算各地区的销售额总和，再用中文总结统计结果。",
        ds,
        first_tool="group_aggregate",
    )
    print("模型回复：", result.answer)
    print("本轮调用：", [t["tool"] for t in result.trace])
    verified = any(t["tool"] == "group_aggregate" and t["result"].get("status") == "ok" for t in result.trace)
    if verified:
        print("[PASS] DeepSeek API → Function Calling → Pandas → 模型回复")
        return 0
    print("[FAIL] 尚未完成真实工具调用链路。请检查模型名、API Key、配额、网络和工具返回信息。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
