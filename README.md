# Data Analysis Agent

一个基于 DeepSeek 和 Function Calling 实现的数据分析 Agent，支持通过自然语言完成常见的数据处理、统计分析和可视化任务。

项目使用 Streamlit 构建交互界面，Pandas 负责数据计算，Matplotlib 和 Seaborn 负责绘图。用户可以上传 CSV 或 Excel 文件，通过对话完成数据探索、分组统计、时间趋势分析以及简单的数据清洗。

## 功能介绍

目前支持以下功能：

- **数据上传**：读取 CSV、TSV 和 Excel 文件，查看字段类型、缺失值及数据概况。
- **探索性分析**：描述性统计、频数统计、条件筛选、分组聚合和相关性分析。
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
│   ├── schemas.py          # 参数校验和工具定义
│   ├── dataset.py          # 数据读取与管理
│   ├── cleaning.py         # 数据清洗与确认
│   ├── trends.py           # 时间趋势分析
│   └── visualization.py    # 图表生成
├── scripts/
│   └── check_deepseek.py   # DeepSeek API 测试
├── sample_data/
│   └── sales_demo.csv
├── tests/                  # 自动化测试
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

**数据清洗**

> 使用中位数填充客户满意度这一列的缺失值。

对于数据清洗任务，Agent 会先生成处理预览。用户确认后才会修改当前数据，也可以取消操作或恢复原始数据。

## 测试

项目使用 pytest 进行自动化测试，覆盖数据处理、趋势计算、参数校验、工具调用和数据清洗等功能。

```bash
python -m pytest -q
```

当前版本包含 24 项自动化测试，已在本地测试通过。

如果需要单独检查 DeepSeek API 和 Function Calling，可以运行：

```bash
python scripts/check_deepseek.py
```

此命令会发起实际 API 请求。

## 说明

这是一个面向单用户、本地数据分析场景的个人项目，主要用于探索大语言模型与传统 Python 数据分析工具的结合方式。

目前更适合常见的结构化数据分析任务，不涉及复杂的数据仓库、多用户权限管理或生产环境部署。模型也可能选择不合适的工具或参数，因此分析结果仍需要根据实际数据进行核对。

项目参考了 [quguolin/llm-data-agent](https://github.com/quguolin/llm-data-agent) 的产品目标和部分架构思路，采用独立实现的轻量化方案，并非直接 Fork 原仓库。相关说明见 [NOTICE.md](NOTICE.md)。