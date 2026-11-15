# 技术设计、简历答辩重点

## 来源与取舍

参考：https://github.com/quguolin/llm-data-agent 。本项目独立实现，用 **OpenAI SDK 兼容 DeepSeek 的 Function Calling** 实现单 Agent 工具调用循环。支持受限工具、上传、统计、图表、清洗确认、多维汇总、业务 Profile 和运行观测。不直接复制/修改上游源码，也没有 Git Fork 提交历史。

## 主要模块

| 模块 | 用途 | 技术要点 |
| --- | --- | --- |
| `core/agent.py` | 构造提示词、模型请求与工具调用循环 | DeepSeek OpenAI 兼容 API，工具调用回传，多轮上限，日志 trace |
| `core/schemas.py` | 固定工具白名单，生成 JSON Schema | Pydantic `extra=forbid`、类型约束、枚举、关联字段验证 |
| `core/tools.py` | 运行 Pandas 计算 | 统计工具与清洗预览统一入口，返回结构化 JSON，拒绝未知工具 |
| `core/aggregation.py` | 多维多指标汇总 | 0–3 个维度、1–6 个指标、显式缺失口径、Top N 截断说明 |
| `core/profiles.py` | 业务口径提示 | 规则识别销售字段组合，可手动切换；不训练模型、不自动建立指标映射 |
| `core/trends.py` | 日期聚合 | `pd.to_datetime` 解析、`Period` 时间粒度、时间排序、无效日期计数、不凭空补零 |
| `core/cleaning.py` | 提案确认 | 模型只能提出 `clean_preview`；数据版本号及预览统计校验后，用户确认才改变数据 |
| `core/visualization.py` | 输出图表 | Matplotlib + Seaborn 只接受受校验的图表规格，在本地绘图 |
| `app_streamlit.py` | 交互展示 | 数据上传、会话、结果复核、图表导出、清洗确认/恢复 |

## 核心数据流程

```text
用户自然语言 + 当前 Schema
    ↓
DeepSeek.chat.completions.create(tools=function_specs())
    ↓
选择工具（白名单）和参数 JSON
    ↓
Pydantic 验证 → Pandas 统计
    ↓
工具结果返回模型做文字解释，图表则本地绘制
    ↓
Streamlit 显示数据、工具记录和结论
```

## 自然语言清洗安全设计

`clean_preview` 是只读工具。Agent 返回的最后一个成功清洗预览会生成 `{revision, arguments, preview}` 待确认提案。用户在对话区点击确认后才执行 `confirm_proposal`。确认前复核数据版本号、Pydantic 参数和清洗前后行数/缺失数；只要上传、手工清洗或恢复原始数据导致版本变化，旧提案就不再有效。执行后清空旧分析记录，防止在已变更数据上展示陈旧统计结论。

## 时间趋势不是预测

`time_trend` 工具支持日/周/月/季度的分组 `count/sum/mean/median`，日期无效行计数，返回严格按时间升序排序的聚合表及折线图。默认展示最近 60 个周期，最大 120 个周期。未出现的日期/周期不补零；缺失值的均值/总和遵循 Pandas 的有效值规则（全缺失组的总和保留为空）。这是**描述性时间趋势**，并非机器学习预测。

## 安全与隐私边界

- 没有任意代码执行工具，API 只能选择预置的 Pandas 运算；并非生产级沙箱。
- 模型看到 Schema 和工具聚合结果，而非文件中的原始数据行。但聚合结果/频数表可能仍暴露个人信息，不可作为完整匿名化方案。
- 输入有 15MB/25 万行/150 列限制；部分操作如复杂对象列可能有耗时或内存问题。
- 不实现用户鉴权/多租户隔离，且模型可能选错工具；不应将该演示直接放到公网或使用敏感业务数据。

## 测试与边界

工具与 Agent 循环通过 pytest 验证；模型响应使用 Fake / Scripted Client，覆盖错误反馈后修正、调用上限、Token 缺失与部分返回，不产生 API 费用。Streamlit AppTest 覆盖多维汇总、组合筛选、数据恢复后旧结果失效，以及运行统计渲染。AppTest 不等同于浏览器视觉验收，也不能证明真实模型路由正确。`scripts/check_deepseek.py` 可单独验证实际 API 连通性；新增功能的真实模型验收问题见 `docs/ACCEPTANCE.md`。

## 多维汇总与筛选

新增 `aggregate_table` 工具。空分组列表表示整体汇总；`row_count` 是行数，`count` 是非缺失值数量，`nunique` 是非缺失去重数，`sum` 对数量字段求和。全缺失金额及空集金额求和返回空值；分组缺失值保留。返回每个指标的口径与全部筛选后数据的有效行数，Top N 只截断展示，不改变有效行数的统计范围。

`where` 兼容原单条件格式，也支持 `{logic: and/or, conditions: [...]}`。最多 8 个条件，不允许嵌套。日期字符串需显式指定 `value_type=datetime`，按 UTC 比较，无法解析的日期和缺失值不参与比较（包括不等于）；`is_null/not_null` 检查原始缺失。汇总、趋势、频数和绘图共用 `Dataset.filter_df`，避免不同工具筛选口径漂移。

## 业务 Profile 与观测

自动模式仅在金额字段与订单字段同时命中预定义名称时加载销售规则；用户可强制通用/销售模式，界面显示实际规则及选择原因。销售规则要求区分订单数求和、订单编号去重与行数，不能确定数据粒度时先询问。规则是提示词约束，不能替代结果评测。

`AgentResult.stats` 保存轮次级耗时、模型请求次数/耗时、工具执行数、工具错误数、停止原因与 Token 用量；每条 trace 含工具耗时。Token 仅累计完整返回 prompt/completion/total 三项的响应，并标记 complete/partial/unavailable；未返回时为 null，避免把未知误报为零。模型异常保留此前已完成的 trace 与用量。耗时不含 UI 绘图，模型请求次数按 SDK 调用计数（内部重试不单独计次），JSON 可由用户下载复盘。

## 面试答题提示

**为什么不直接让模型写 Python？** 因为任意代码执行、复杂依赖、注入及结果可复核性都有额外风险。采用工具白名单，模型负责意图识别，Pandas 负责确定性计算。

**为什么要 Pydantic？** 可以限制字段、枚举范围、数值区间和关联条件，并把错误安全地反馈给模型做参数调整；减少非法工具调用。

**怎么评价质量？** 对确定性工具写单测，给模型工具路由准备任务集；分别统计工具调用成功率、任务完成率、时间聚合准确率、幻觉率与响应延迟。当前并没有大样本端到端评测数据，简历不要编造百分比。
