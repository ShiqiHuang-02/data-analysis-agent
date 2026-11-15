# 评测体系设计与使用

本目录 `evals/` 为 Data Analysis Agent 提供一套**端到端、可复现**的评测：把自然语言问题喂给真实 Agent，用确定性工具结果作为黄金答案，自动统计工具路由准确率、结果准确率、完成率、幻觉率（启发式）、延迟与成本。

## 为什么要有评测

Agent 的"聪明"体现在两个环节：**（1）选对工具和参数（路由）**，**（2）数值答对**。Pandas 计算是确定性的（已被 45 个 pytest 覆盖），但模型选错工具或参数时，Pandas 会算出一个"逻辑上正确、但答非所问"的结果。单靠单测测不到这条链路，所以需要端到端评测。

## 任务集设计

`evals/tasks.py` 里 41 条任务覆盖全部 9 个工具，每条含：

| 字段 | 含义 |
| --- | --- |
| `question` | 自然语言问题 |
| `expected_tool` | 期望调用的工具 |
| `golden_args` | 能确定性算出正确答案的工具参数 |

黄金结果 = `run_tool(ds, expected_tool, golden_args)`，在运行时本地计算，不需要手写数值。

分布：overview 3、missing 1、value_counts 4、group_aggregate 6、correlation 2、time_trend 5、aggregate_table 10、make_chart 6、clean_preview 4。

### 评分规则

- **路由准确（routing_ok）**：Agent 的工具调用序列里出现过 `expected_tool`。
- **结果准确（result_ok）**：Agent 对 `expected_tool` 的某次成功调用，其返回结果与黄金结果**语义等价**。比较时会对表格做行排序、对浮点四舍五入到 4 位小数；`aggregate_table` 比较的是**指标数值**而非模型自定的指标名（同一个"销售额求和"起什么名字不影响正确性）。

## 指标定义

| 指标 | 定义 |
| --- | --- |
| 工具路由准确率 | 期望工具被调用的任务占比 |
| 结果准确率 | 调出的结果与黄金结果等价的任务占比 |
| 任务完成率 | 路由且结果都正确的任务占比（最严的口径） |
| 幻觉率（启发式） | 回答中出现「工具结果里没有的数字」的任务占比 |
| 平均耗时 / Token / 成本 | 端到端延迟、用量与按单价估算的费用 |

## 使用方法

### 1. 自检（不花钱，应先跑）

```bash
python -m evals.runner --mock
```

mock 模式直接把黄金工具调用注入评分器，**应得 100%**。它只验证「任务集 + 评分器」自洽，不代表模型表现。跑出来如果不是 100%，说明任务集或评分器有 bug。

### 2. 真实评测（调用 API，产生费用）

```bash
# 先在前 10 条上试跑，确认 API 正常
python -m evals.runner --limit 10

# 全量 41 条
python -m evals.runner

# 换模型对比（模型选型 vs 成本）
python -m evals.runner --model deepseek-chat
python -m evals.runner --model deepseek-flash
```

结果写入 `evals/results/report_<时间戳>.json`、`latest.json`、`latest.csv`（该目录已 gitignore，不提交）。报告只保存聚合统计与工具参数/聚合结果，**不含原始数据行**。

### 3. 换自己的数据

```bash
python -m evals.runner --data path/to/your.csv
```

任务集里的字段名（地区/渠道/订单数/销售额/客户满意度/是否复购/日期）是照着 `sales_demo.csv` 写的，换数据后需要同步改写 `evals/tasks.py` 的 `golden_args`。

## 局限（如实声明）

- **不是大样本**：41 条任务、单一数据源，只能反映「在这份销售数据上的路由表现」，不能推广到任意数据。
- **路由不稳定**：大模型对同一问题可能给出不同工具选择，单次结果有噪声；应多次运行取平均，并区分「选错工具（路由错）」与「参数错（结果错）」。
- **幻觉率是启发式**：只做「数字是否出现在工具结果里」的粗查，有假阳性（如回答复述了问题里的数字）和假阴性（如编造了工具结果里恰好也有的数字）。
- **成本是估算**：单价来自 `evals/runner.py` 顶部的 `PRICE_CNY_PER_1M`（缓存未命中价），未计缓存命中折扣与峰谷差异，运行前请按 [DeepSeek 官方价格页](https://api-docs.deepseek.com/quick_start/pricing) 核对。
- **不测清洗的"确认"链路**：`clean_preview` 是只读预览，真实数据变更需用户在 UI 确认，这部分仍由 `tests/test_completion.py` 的单测覆盖。

## 下一步（可选）

- 增加**对抗/歧义任务**（如"订单数"到底该 sum 还是 count），验证 Agent 是否会先询问口径。
- 增加**多轮会话**任务，测上下文记忆与追问。
- 对失败的案例做**归因**：把「选错工具」单独拆出来，指导系统提示词与工具描述迭代（对应 `docs/DESIGN.md` 的"怎么评价质量"）。
