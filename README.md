# Data Analysis Agent

一个基于 DeepSeek 和 Function Calling 实现的数据分析 Agent，支持通过自然语言完成常见的数据处理、统计分析和可视化任务。

项目使用 Streamlit 构建交互界面，Pandas 负责数据计算，Matplotlib 和 Seaborn 负责绘图。用户可以上传 CSV 或 Excel 文件，通过对话完成数据探索、分组统计、时间趋势分析以及简单的数据清洗。

## 在线 Demo

本项目可直接部署到 [Streamlit Community Cloud](https://streamlit.io/cloud)（免费，连接 GitHub 仓库，无需服务器）。

[![部署到 Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)

**部署步骤（约 2 分钟）：**

1. 打开 [share.streamlit.io](https://share.streamlit.io/)，用 GitHub 账号登录。
2. 点 **New app**，仓库选 `ShiqiHuang-02/data-analysis-agent`，分支 `main`，主文件填 `app_streamlit.py`。
3. 点 **Deploy**，等构建完成即可得到公网链接。

**启用自然语言对话（可选）：** 进入应用的 **Settings → Secrets**，粘贴一条 secret：

```toml
DEEPSEEK_API_KEY = "你的 DeepSeek Key"
DEEPSEEK_MODEL = "deepseek-chat"
```

不配置 Key 也能用：内置销售示例数据，以及离线分析、多维汇总、数据清洗等功能均可用；配置后即可使用 Agent 自然语言对话。

> 云端已通过 `packages.txt` 安装中文字体（Noto CJK），图表中文标签正常显示。

## 功能介绍

目前支持以下功能：

- **数据上传**：读取 CSV、TSV 和 Excel 文件，查看字段类型、缺失值及数据概况。
- **探索性分析**：描述性统计、频数统计、条件筛选、分组聚合和相关性分析。
- **多维多指标汇总**：最多按 3 个字段分组、同时计算 6 个指标，明确区分行数、非缺失数、去重数、数量求和；支持整体汇总和 Top N。
- **组合筛选**：最多 8 个条件的 AND / OR 组合，支持显式日期解析；可用于汇总、频数、趋势和图表。
- **业务口径提示**：自动识别或手动选择销售场景，提示订单数量、订单编号、记录行数等口径区别；规则可在侧栏查看。
- **运行观测**：展示每轮耗时、工具耗时、模型请求次数、Token 用量与估算成本，支持运行记录 JSON 及 Langfuse / OpenTelemetry 格式追踪导出。
- **SQL 查询**：当固定工具无法表达时（窗口函数、CASE WHEN、子查询等），可提交只读 SQL（内存 SQLite，仅单条 SELECT，结果有上限）。
- **数据可视化**：根据分析需求生成柱状图、折线图、直方图、散点图和箱线图。
- **趋势分析**：识别日期字段，支持按日、周、月、季度聚合数据并生成趋势图。
- **数据清洗**：通过自然语言提出清洗需求，预览处理结果，经用户确认后应用修改。
- **结果导出**：下载分析结果、清洗后的数据及生成的图表。

项目内置了一份模拟销售数据，可以直接用于测试。

## 技术实现

主要使用的技术：

- Python
- DeepSeek API / Function Calling
- Pandas
- Pydantic
- Matplotlib / Seaborn
- Streamlit

### Agent 工作流程

Agent 采用基于工具调用的实现方式，模型负责理解用户需求和选择工具，具体的数据处理任务由本地 Python 函数完成。

```text
用户输入
   ↓
Streamlit 交互界面
   ↓
DeepSeek API
   ↓
Function Calling（工具选择与参数生成）
   ↓
Pydantic 参数校验
   ↓
Pandas 数据处理 / 统计分析
   ↓
返回分析结果
   ↓
Streamlit 展示表格、图表及文字回复
```

目前将数据分析能力封装为预定义工具，由 Agent 根据用户问题决定调用哪些工具及其参数。

没有开放任意 Python 代码执行接口，模型不能直接执行 `exec()` 或 `eval()`。数据清洗采用预览和确认机制，避免在用户不知情的情况下修改数据。

### 项目结构

```text
data-analysis-agent/
├── app_streamlit.py
├── core/
│   ├── agent.py            # 模型调用与工具执行流程
│   ├── tools.py            # Pandas 数据分析工具
│   ├── aggregation.py      # 多维多指标汇总与统计口径
│   ├── profiles.py         # 业务场景识别与动态提示词
│   ├── schemas.py          # 参数校验和工具定义
│   ├── dataset.py          # 数据读取与管理
│   ├── cleaning.py         # 数据清洗与确认
│   ├── trends.py           # 时间趋势分析
│   ├── visualization.py    # 图表生成
│   ├── sql.py              # 只读 SQL 工具
│   └── observability.py    # 成本估算与追踪导出
├── scripts/
│   └── check_deepseek.py   # DeepSeek API 测试
├── sample_data/
│   └── sales_demo.csv
├── tests/                  # 自动化测试
├── evals/                  # 端到端评测任务集与评分器
├── docs/                   # 设计说明与验收文档
├── .env.example
├── requirements.txt
├── LICENSE
└── NOTICE.md
```

## 快速开始

### 1. 安装依赖

项目已在本地 Python 3.14 环境下运行。

```bash
git clone https://github.com/<your-username>/data-analysis-agent.git
cd data-analysis-agent

python3 -m venv .venv
source .venv/bin/activate

python -m pip install -r requirements.txt
```

Windows 用户需要使用对应的虚拟环境激活命令。

### 2. 配置 DeepSeek API

在项目根目录创建 `.env` 文件：

```bash
cp .env.example .env
```

打开 `.env`，填写自己的 DeepSeek API Key，并按照 `.env.example` 配置模型名称及 API 地址。

API Key 可以从 [DeepSeek 开放平台](https://platform.deepseek.com/) 获取。

不要将包含真实密钥的 `.env` 文件提交到 GitHub。

### 3. 启动项目

```bash
python -m streamlit run app_streamlit.py
```

启动后访问：

http://localhost:8501

如果没有配置 API Key，也可以使用离线分析和手工数据清洗功能。

## 使用示例

可以上传自己的数据，也可以使用项目提供的 `sales_demo.csv`。

以下是一些可以直接输入的分析指令。

**数据概况**

> 总结这份数据的基本情况，包括字段类型和缺失值情况。

**分组统计**

> 按照地区统计销售额总和，并按照销售额从高到低排序。

**趋势分析**

> 按月汇总销售额，并绘制销售额随时间变化的折线图。

**多维多指标汇总**

> 按地区和渠道，统计销售额总和、订单数总和和平均客户满意度。按销售额总和降序。

**组合筛选**

> 筛选地区为华北且销售额大于 100 的记录，按渠道汇总销售额与订单数。

> 筛选日期在 2026-01-01（含）到 2026-02-01（不含）的记录，统计销售额总和。

无需 API 时可在「多维汇总」页面选择分组、指标和筛选条件。结果附有指标定义、有效记录数和截断说明，便于与 Agent 的执行结果交叉核对。订单数列求和与数据行数并不相同；此示例不把每行视为一张订单。

**数据清洗**

> 使用中位数填充客户满意度这一列的缺失值。

对于数据清洗任务，Agent 会先生成处理预览。用户确认后才会修改当前数据，也可以取消操作或恢复原始数据。

## 测试

项目使用 pytest 进行自动化测试，覆盖数据处理、趋势计算、参数校验、工具调用和数据清洗等功能。

```bash
python -m pytest -q
```

测试包含工具计算、组合筛选、日期边界、缺失值口径、模拟模型错误反馈与修正、Token 缺失/部分返回以及 Streamlit 离线交互。Agent 流程测试使用模拟客户端，不代表真实模型的任务准确率。

如果需要单独检查 DeepSeek API 和 Function Calling，可以运行：

```bash
python scripts/check_deepseek.py
```

此命令会发起实际 API 请求。

## 评测

项目附带一套端到端评测（`evals/`），用 44 条自然语言任务自动统计 Agent 的**工具路由准确率、结果准确率、任务完成率、幻觉率（启发式）、延迟与 Token 成本**。

```bash
python -m evals.runner --mock        # 自检评分器，不调用 API，应得 100%
python -m evals.runner --limit 10    # 真实 API 试跑前 10 条
python -m evals.runner               # 全量 44 条
python -m evals.runner --repeat 3    # 重复 3 次取平均，降低路由随机性
```

真实评测会调用 DeepSeek API 并产生费用；结果写入 `evals/results/`（已 gitignore），只保存聚合统计与工具参数/聚合结果，不含原始数据行。评测方法、指标定义与局限见 [docs/EVAL.md](docs/EVAL.md)。

## 说明

这是一个面向单用户、本地数据分析场景的个人项目，主要用于探索大语言模型与传统 Python 数据分析工具的结合方式。

目前更适合常见的结构化数据分析任务，不涉及复杂的数据仓库、多用户权限管理或生产环境部署。模型也可能选择不合适的工具或参数，因此分析结果仍需要根据实际数据进行核对。

业务 Profile 是可查看的规则与提示词，不是模型训练，也不自动保证业务口径正确。当前筛选支持单层 AND / OR，不支持混合嵌套逻辑。Token 未返回时显示“未返回”，仅部分请求有用量时明确标注；不估算费用。运行记录由用户主动下载，不自动写入磁盘。

项目参考了 [quguolin/llm-data-agent](https://github.com/quguolin/llm-data-agent) 的产品目标和部分架构思路，采用独立实现的轻量化方案，并非直接 Fork 原仓库。相关说明见 [NOTICE.md](NOTICE.md)。
