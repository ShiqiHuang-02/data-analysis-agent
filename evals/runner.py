"""Run the agent end-to-end evaluation and report routing / result / cost metrics.

Usage:
    python -m evals.runner --mock              # 自检评分器（不调用 API，应得 100%）
    python -m evals.runner --limit 10          # 真实 API 跑前 10 条任务
    python -m evals.runner --model deepseek-chat --data my_data.csv

真实模式会调用 DeepSeek API 并产生费用；结果只保存聚合统计与工具参数/聚合结果，
不包含原始数据行（与 Agent 运行记录同一口径）。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from core.agent import run_agent
from core.observability import estimate_cost
from core.tools import run_tool
from evals.tasks import TASKS, load_dataset

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _round(value: Any) -> Any:
    """Recursively round floats so results from identical computations compare equal."""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, 4)
    if isinstance(value, dict):
        return {str(k): _round(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round(v) for v in value]
    return value


def _sorted_table(table: list[dict]) -> list[dict]:
    """Sort table rows into a canonical order (row/column order independent)."""
    rows = [_round(row) for row in table]
    rows.sort(key=lambda r: json.dumps(r, ensure_ascii=False, sort_keys=True, default=str))
    return rows


def _canonical_aggregate(table: list[dict], group_by: list[str], result: dict) -> dict:
    """Aggregate_table compares metric *values* (not the model-chosen names)."""
    group_rows = []
    value_columns = []
    if table:
        keys = {k for row in table for k in row}
        value_keys = [k for k in keys if k not in set(group_by)]
        for row in table:
            group_rows.append(_round({g: row.get(g) for g in group_by}))
        for vk in value_keys:
            vals = [_round(row.get(vk)) for row in table]
            vals.sort(key=lambda v: json.dumps(v, ensure_ascii=False, sort_keys=True, default=str))
            value_columns.append(vals)
    value_columns.sort(key=lambda v: json.dumps(v, ensure_ascii=False, sort_keys=True, default=str))
    return {"group_by": sorted(group_by),
            "group_rows": _sorted_table(group_rows),
            "value_columns": value_columns,
            "filtered_rows": result.get("filtered_rows"),
            "total_groups": result.get("total_groups")}


def _canonical_sql(table: list[dict], result: dict) -> dict:
    """SQL 结果比较数值行（忽略模型自定的列别名与列顺序）。"""
    rows = []
    for row in table:
        vals = [_round(v) for v in row.values()]
        vals.sort(key=lambda v: json.dumps(v, ensure_ascii=False, sort_keys=True, default=str))
        rows.append(vals)
    rows.sort(key=lambda r: json.dumps(r, ensure_ascii=False, sort_keys=True, default=str))
    return {"row_count": result.get("row_count"), "rows": rows,
            "truncated": result.get("truncated")}


def canonicalize(result: dict, tool: str, group_by: list[str] | None = None) -> dict:
    """Reduce a tool result to the part that determines correctness."""
    if not isinstance(result, dict) or result.get("status") != "ok":
        return {"status": result.get("status", "error") if isinstance(result, dict) else "error"}
    kind = result.get("kind")
    if kind == "overview":
        return {"row_count": result.get("row_count"), "column_count": result.get("column_count"),
                "table": _sorted_table(result.get("table", []))}
    if kind == "clean_preview":
        return {"operation": result.get("operation"), "column": result.get("column"),
                "before_rows": result.get("before_rows"), "after_rows": result.get("after_rows")}
    if kind == "chart":
        spec = result.get("chart_spec") or {}
        return {"kind": spec.get("kind"), "x": spec.get("x"), "y": spec.get("y"),
                "agg": spec.get("agg"), "filtered_rows": result.get("filtered_rows")}
    table = result.get("table")
    if table is None:
        return {"kind": kind, "filtered_rows": result.get("filtered_rows"),
                "row_count": result.get("row_count")}
    if tool == "aggregate_table":
        return _canonical_aggregate(table, group_by or [], result)
    if tool == "sql_query":
        return _canonical_sql(table, result)
    return {"table": _sorted_table(table)}


def results_match(golden: dict, agent: dict, tool: str, group_by: list[str] | None = None) -> bool:
    return canonicalize(golden, tool, group_by) == canonicalize(agent, tool, group_by)


def _numbers_in_trace(trace: list[dict]) -> set[float]:
    found: set[float] = set()
    for entry in trace:
        payload = json.dumps(entry.get("result", {}), ensure_ascii=False, default=str)
        for m in _NUM.findall(payload):
            try:
                found.add(round(float(m), 4))
            except ValueError:
                continue
    return found


def hallucinated_numbers(answer: str, trace: list[dict]) -> list[float]:
    """启发式：回答中的数字若不出现在任何工具结果里，记为可疑（含假阳性）。"""
    if not answer:
        return []
    present = _numbers_in_trace(trace)
    out: list[float] = []
    for m in _NUM.findall(answer):
        try:
            n = round(float(m), 4)
        except ValueError:
            continue
        # 排除年份、以及工具调用次数等小整数，降低误报。
        if n.is_integer() and (1900.0 <= n <= 2100.0 or 0.0 <= n <= 5.0):
            continue
        if n not in present:
            out.append(n)
    return out


def grade_task(task: dict, agent_result: Any, golden: dict) -> dict:
    tools = [t["tool"] for t in agent_result.trace]
    routing_ok = task["expected_tool"] in tools
    result_ok = False
    for entry in agent_result.trace:
        if entry["tool"] == task["expected_tool"] and entry["result"].get("status") == "ok":
            if results_match(golden, entry["result"], task["expected_tool"],
                             task["golden_args"].get("group_by")):
                result_ok = True
                break
    return {
        "id": task["id"], "category": task["category"], "question": task["question"],
        "expected_tool": task["expected_tool"], "tools_called": tools,
        "routing_ok": routing_ok, "result_ok": result_ok,
        "hallucinated": hallucinated_numbers(agent_result.answer, agent_result.trace),
        "stats": agent_result.stats, "answer": agent_result.answer,
    }


def aggregate(scores: list[dict]) -> dict:
    n = max(len(scores), 1)
    routing = sum(s["routing_ok"] for s in scores)
    result = sum(s["result_ok"] for s in scores)
    complete = sum(s["routing_ok"] and s["result_ok"] for s in scores)
    hallucinated = sum(1 for s in scores if s["hallucinated"])
    tokens = {"prompt": 0, "completion": 0, "total": 0}
    cost = 0.0
    latencies: list[float] = []
    model = None
    by_category: dict[str, dict] = {}
    for s in scores:
        st = s["stats"]
        model = model or st.get("model")
        latencies.append(st.get("elapsed_ms", 0.0))
        if st.get("total_tokens") is not None:
            tokens["total"] += st["total_tokens"]
            tokens["prompt"] += st.get("prompt_tokens", 0)
            tokens["completion"] += st.get("completion_tokens", 0)
        c = estimate_cost(model, st.get("prompt_tokens"), st.get("completion_tokens"))
        if c is not None:
            cost += c
        cat = by_category.setdefault(s["category"], {"n": 0, "routing": 0, "result": 0})
        cat["n"] += 1
        cat["routing"] += s["routing_ok"]
        cat["result"] += s["result_ok"]
    category = {c: {"n": v["n"], "routing_accuracy": v["routing"] / v["n"],
                    "result_accuracy": v["result"] / v["n"]}
                for c, v in sorted(by_category.items())}
    return {
        "n_runs": len(scores), "n_unique_tasks": len({s["id"] for s in scores}),
        "model": model,
        "routing_accuracy": routing / n,
        "result_accuracy": result / n,
        "completion_rate": complete / n,
        "hallucination_rate": hallucinated / n,
        "mean_latency_ms": (sum(latencies) / n) if latencies else 0.0,
        "tokens": tokens, "estimated_cost_cny": round(cost, 6),
        "by_category": category,
    }


def build_client():
    from openai import OpenAI
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise SystemExit("未配置 DEEPSEEK_API_KEY。请先在 .env 中设置，或用 --mock 自检。")
    return OpenAI(api_key=key, base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                  timeout=45.0, max_retries=1)


def run_eval(tasks: list[dict], ds, client, model: str, profile_mode: str = "auto") -> list[dict]:
    scores: list[dict] = []
    for i, task in enumerate(tasks, 1):
        golden = run_tool(ds, task["expected_tool"], task["golden_args"])
        agent = run_agent(task["question"], ds, client=client, model=model, profile_mode=profile_mode)
        score = grade_task(task, agent, golden)
        scores.append(score)
        mark = "✓" if score["routing_ok"] else "✗"
        print(f"[{i}/{len(tasks)}] {task['id']} {mark} "
              f"{'/'.join(score['tools_called']) or '（未调用工具）'}  {task['question'][:26]}",
              file=sys.stderr, flush=True)
    return scores


def run_mock(tasks: list[dict], ds) -> list[dict]:
    """自检：直接注入黄金工具调用，验证评分器应给 100%。"""
    scores: list[dict] = []
    for task in tasks:
        golden = run_tool(ds, task["expected_tool"], task["golden_args"])
        trace = [{"tool": task["expected_tool"], "arguments": task["golden_args"],
                  "result": golden, "duration_ms": 1.0}]
        stats = {"model": "mock", "elapsed_ms": 0.0, "prompt_tokens": 0,
                 "completion_tokens": 0, "total_tokens": 0, "usage_status": "complete"}
        scores.append(grade_task(task, SimpleNamespace(answer="", trace=trace, stats=stats), golden))
    return scores


def print_report(scores: list[dict], agg: dict, note: str = "") -> None:
    print("\n== 逐任务结果 ==")
    for s in scores:
        r = "✓" if s["routing_ok"] else "✗"
        v = "✓" if s["result_ok"] else "✗"
        hall = f" 幻觉={s['hallucinated']}" if s["hallucinated"] else ""
        run = f"[run{s['run']}] " if s.get("run") else ""
        print(f"{run}{s['id']:>4} 路由{r} 结果{v}  {s['expected_tool']:<16} "
              f"{'/'.join(s['tools_called']) or '—':<32} {s['question'][:24]}{hall}")
    runs, uniq = agg["n_runs"], agg["n_unique_tasks"]
    task_label = f"{uniq}（运行 {runs} 次）" if runs != uniq else f"{uniq}"
    print("\n== 汇总 ==")
    print(f"任务数        {task_label}")
    print(f"模型          {agg['model']}")
    print(f"工具路由准确率 {agg['routing_accuracy']:.1%}")
    print(f"结果准确率    {agg['result_accuracy']:.1%}")
    print(f"任务完成率    {agg['completion_rate']:.1%}（路由且结果均正确）")
    print(f"幻觉率(启发式) {agg['hallucination_rate']:.1%}")
    print(f"平均耗时      {agg['mean_latency_ms']:.0f} ms")
    print(f"Token         {agg['tokens']['total']:,}（输入 {agg['tokens']['prompt']:,} / 输出 {agg['tokens']['completion']:,}）")
    print(f"估算成本      ¥{agg['estimated_cost_cny']:.4f}（按配置单价，缓存命中未计入，见 docs/EVAL.md）")
    print("\n== 按工具分类 ==")
    for c, v in agg["by_category"].items():
        print(f"{c:<16} n={v['n']:>2}  路由 {v['routing_accuracy']:.0%}  结果 {v['result_accuracy']:.0%}")
    if note:
        print(f"\n{note}")


def write_report(scores: list[dict], agg: dict, dataset: str, out_dir: Path | None) -> Path | None:
    if out_dir is None:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {"generated_at": datetime.now(timezone.utc).isoformat(), "dataset": dataset,
               "aggregate": agg, "tasks": scores}
    (out_dir / f"report_{stamp}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    latest = out_dir / "latest.json"
    latest.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    csv_path = out_dir / "latest.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "category", "question", "expected_tool", "tools_called",
                    "routing_ok", "result_ok", "hallucinated", "elapsed_ms", "total_tokens"])
        for s in scores:
            w.writerow([s["id"], s["category"], s["question"], s["expected_tool"],
                        "|".join(s["tools_called"]), s["routing_ok"], s["result_ok"],
                        len(s["hallucinated"]), s["stats"].get("elapsed_ms"),
                        s["stats"].get("total_tokens")])
    return latest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Data Analysis Agent 端到端评测")
    p.add_argument("--mock", action="store_true", help="自检评分器（不调用 API，应得满分）")
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 条任务（0=全部）")
    p.add_argument("--model", default=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
    p.add_argument("--profile-mode", default="auto", choices=["auto", "general", "sales"])
    p.add_argument("--data", default=None, help="数据文件路径（默认 sample_data/sales_demo.csv）")
    p.add_argument("--repeat", type=int, default=1, help="重复运行次数（>1 时多次运行取平均，降低路由随机性）")
    p.add_argument("--output", default=None, help="结果输出目录（默认 evals/results/）")
    args = p.parse_args(argv)

    tasks = TASKS[: args.limit] if args.limit else TASKS
    ds = load_dataset(args.data)

    if args.mock:
        scores = run_mock(tasks, ds)
        agg = aggregate(scores)
        note = "mock 模式：直接注入黄金工具调用，只验证评分器与任务集自洽，不代表真实模型表现。"
    else:
        client = build_client()
        scores = []
        for run_i in range(args.repeat):
            if args.repeat > 1:
                print(f"\n== 第 {run_i + 1}/{args.repeat} 轮 ==", file=sys.stderr)
            batch = run_eval(tasks, ds, client, args.model, args.profile_mode)
            for s in batch:
                s["run"] = run_i + 1
            scores.extend(batch)
        agg = aggregate(scores)
        note = "真实模型结果；工具路由可能不稳定，建议多次运行取平均，不要把单次结果当成稳定结论。"

    print_report(scores, agg, note)
    out_dir = Path(args.output) if args.output else (Path("evals") / "results")
    write_report(scores, agg, ds.filename, out_dir)
    if args.output or not args.mock:
        print(f"\n结果已写入 {out_dir}/latest.json 与 latest.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
