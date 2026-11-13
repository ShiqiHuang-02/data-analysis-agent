# 技术设计、简历答辩重点

## 来源与取舍

参考：https://github.com/quguolin/llm-data-agent 。本项目独立实现，用 **OpenAI SDK 兼容 DeepSeek 的 Function Calling** 替代复杂的多框架编排。仅保留个人项目必需的对话、受限工具、上传、统计、图表、清洗确认、测试。不直接复制/修改上游源码，也没有 Git Fork 提交历史。

## 主要模块

| 模块 | 用途 | 技术要点 |
| --- | --- | --- |
| `core/agent.py` | 构造提示词、模型请求与工具调用循环 | DeepSeek OpenAI 兼容 API，工具调用回传，多轮上限，日志 trace |
| `core/schemas.py` | 固定工具白名单，生成 JSON Schema | Pydantic `extra=forbid`、类型约束、枚举、关联字段验证 |
| `core/tools.py` | 运行 Pandas 计算 | 统计工具与清洗预览统一入口，返回结构化 JSON，拒绝未知工具 |
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

在构建环境验证 `24 passed`。其中 Agent 真实 Function Calling 的两轮交互在本地使用 Fake Client，避免产生 API 费用。提供 `scripts/check_deepseek.py` 来验证用户自己配置的 API Key 和端到端调用；构建环境没有 Key，且未安装 Streamlit、无法联网安装，因此浏览器尚未通过人工验收。

## 面试答题提示

**为什么不直接让模型写 Python？** 因为任意代码执行、复杂依赖、注入及结果可复核性都有额外风险。采用工具白名单，模型负责意图识别，Pandas 负责确定性计算。

**为什么要 Pydantic？** 可以限制字段、枚举范围、数值区间和关联条件，并把错误安全地反馈给模型做参数调整；减少非法工具调用。

**怎么评价质量？** 对确定性工具写单测，给模型工具路由准备任务集；分别统计工具调用成功率、任务完成率、时间聚合准确率、幻觉率与响应延迟。当前并没有大样本端到端评测数据，简历不要编造百分比。
