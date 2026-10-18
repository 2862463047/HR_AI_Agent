一个由 **LangGraph** 编排、由 **通义千问** 驱动的端到端招聘助理 Agent。从收集岗位需求开始，自动生成职位描述、在 LinkedIn 上检索和联系候选人、分析 CV、生成面试题，并在关键环节（JD 定稿、面试题定稿）挂起工作流等待人工确认——把"人机协作"作为流程的一等公民。

## 一、核心价值

| 传统招聘的痛点                                 | 本项目的解法                                                 |
| ---------------------------------------------- | ------------------------------------------------------------ |
| 岗位需求靠 HR 反复对齐、JD 手写耗时            | **requirements_gathering** 结构化采集 → **generate_job_desc** 一键成稿 |
| LinkedIn 候选人筛选依赖人工搜索，效率低        | **search_linkedin_candidates** 通过 Serper 走 Google 搜 LinkedIn，自动抓取目标画像 |
| Outreach 消息千篇一律，回复率低                | **send_linkedin_message** 由 LLM 结合 JD 现场生成**个性化触达文案** |
| CV 初筛主观且标准不一，耗时长                  | **analyze_cv** 用统一 prompt 打三项分（技能匹配 / 经验 / 总体适配），输出 0–10 分 + 结构化反馈 |
| 面试题准备费时，且质量波动大                   | **prepare_interview** 按"技术 4 + 经验 3 + 问题解决 3"结构生成 10 题 |
| AI 全自动决策风险高（误发拒绝信 / 误发 offer） | **NodeInterrupt 人工闸门**——JD 与面试题**必须人工审批**才能往下走；CV 分数阈值 `7.0` 决定面试/拒绝 |
| 多轮流程易失忆                                 | **MemorySaver + thread_id** 持久化整条招聘会话               |

------

## 二、技术架构

### 2.1 技术栈

| 层            | 组件                                                         | 作用                                                        |
| ------------- | ------------------------------------------------------------ | ----------------------------------------------------------- |
| 流程编排      | **LangGraph** (`StateGraph` + `NodeInterrupt` + `MemorySaver`) | 定义 5 节点 + 3 条件边，节点级**人工断点**                  |
| LLM           | **ChatOpenAI**（OpenAI 兼容协议）接入 **DashScope** 通义千问 `qwen3.8-flash` | 生成 JD、Outreach 文案、CV 打分、面试题                     |
| 外部搜索      | **LangChain GoogleSerperAPIWrapper**                         | `site:linkedin.com/in/ <职位> <地点> <技能>` 定向检索候选人 |
| LinkedIn 集成 | **LinkedIn Voyager 内部 API**（Cookie 认证）                 | 拉取候选人画像、发送站内消息                                |
| 数据模型      | **Pydantic** (`JobRequirements` / `CandidateProfile` / `RecruitmentState`) | 强类型约束状态与结构化输出                                  |
| 状态持久化    | `MemorySaver` + `configurable.thread_id`                     | 会话级检查点，可断点续跑                                    |
| 环境配置      | **python-dotenv**                                            | 密钥集中管理                                                |
| 可视化        | `graph.get_graph().draw_mermaid_png()`                       | 图结构可视化                                                |

### 2.2 工作流拓扑

### 2.3 节点职责

| 节点                     | 输入                                 | 输出                                      | 说明                                                         |
| ------------------------ | ------------------------------------ | ----------------------------------------- | ------------------------------------------------------------ |
| `requirements_gathering` | 首次交互                             | `job_requirements`, `phase`               | 12 项结构化提问（职位名、公司描述、职责、资质、福利、面试日期、国家、年限、LinkedIn URL、技能、薪资等） |
| `generate_job_desc`      | `job_requirements`                   | **NodeInterrupt**（返回 JD 供人工审阅）   | 按 6 段式结构产出 JD，**直接 raise NodeInterrupt**           |
| `linkedin_process`       | JD + `linkedin_profiles`             | `candidates`, `linkedin_process_complete` | 若提供 URL → 逐个触达；否则 Serper 搜索前 5 名 → 逐个抓取画像 → LLM 个性化消息发送 |
| `analyze_cv`             | `candidates[-1]`, `job_requirements` | 更新候选人 `cv_score / status / feedback` | LLM 三维打分，正则解析 `Overall Score: X.X/10`，≥7.0 判 `approved` 否则 `rejected`，结果消息回写 LinkedIn |
| `prepare_interview`      | `job_requirements`                   | **NodeInterrupt**（返回 10 道面试题）     | 结构化生成 10 题：技术 4 + 经验 3 + 问题解决 3               |

### 2.4 关键设计

**① 人工闸门（Human-in-the-Loop）**

python

```
raise NodeInterrupt(f"Please review the generated job description:\n\n{response.content}")
```

配合 `workflow.compile(interrupt_before=[...])` 使用。运维侧通过 `update_state` 更新审批标志后 `stream(None, config)` 续跑：

python

```
recruitment_workflow.update_state(config, {"job_description_approved": True})
for event in recruitment_workflow.stream(None, config, stream_mode="values"): ...
```

**② 状态机式路由**

python

```
def route_after_job_desc(state):
    return "linkedin_process" if state.job_description_approved else "generate_job_desc"

def route_after_cv(state):
    if not state.candidates:                return END
    return "prepare_interview" if state.candidates[-1].status == "approved" else END
```

**③ 候选人生命周期**

text

```
new → contacted → cv_received → approved / rejected
```

由 `CandidateProfile.status` 字段贯穿，`cv_score` 阈值 7.0 为二元分流点。

------

## 三、程序功能说明

### 3.1 结构化岗位需求采集

`requirements_gathering` 以交互式 `input()` 一次性采集 12 个字段并实例化 `JobRequirements`：

| 字段                         | 类型                | 示例                                     |
| ---------------------------- | ------------------- | ---------------------------------------- |
| `title`                      | str                 | `Senior Backend Engineer`                |
| `company_description`        | str                 | `AI 社交发布 SaaS`                       |
| `job_requirements`           | List[str]           | `Python, FastAPI, PostgreSQL`            |
| `candidate_responsibilities` | List[str]           | `设计 REST API…`                         |
| `candidate_qualifications`   | List[str]           | `5 年以上后端经验…`                      |
| `company_benefits`           | List[str]           | `远程、股票期权…`                        |
| `interview_date`             | datetime            | `2026-11-01`                             |
| `preferred_country`          | str                 | `China`                                  |
| `years_experience`           | int                 | `5`                                      |
| `linkedin_profiles`          | Optional[List[str]] | `["https://www.linkedin.com/in/xxx", …]` |
| `skills_required`            | List[str]           | `Python, Redis, Docker`                  |
| `salary_range`               | str                 | `30k–50k / month`                        |

### 3.2 JD 生成（含人工审批）

- Prompt 明确要求 6 段式：**About the Company → Role Overview → Responsibilities → Qualifications → Benefits → Location**。
- 生成后立即 `NodeInterrupt` 中断，交由外部观察者审阅。
- 只有把 `job_description_approved` 置为 `True` 才会走 `linkedin_process`；否则 `route_after_job_desc` 会**原地重进** `generate_job_desc`，等价于"打回重写"。

### 3.3 LinkedIn 候选人触达

**路径 A：提供 URL（精准触达）**

text

```
for url in job_requirements.linkedin_profiles:
    画像 → LLM 生成消息 → POST /voyager/api/messaging/conversations
```

**路径 B：Serper 搜索（扩大漏斗）**

text

```
site:linkedin.com/in/ {title} {country} {skills}
→ 抓前 5 条 organic 结果
→ 逐个 get_linkedin_profile
→ 逐个 send_linkedin_message
```

每条候选人记录包含 `source = "direct" | "search"`、`message_sent`、`status = "contacted"`。

### 3.4 CV 打分与分流

- 评分维度：**Skills Match / Experience / Overall Fit**。
- 输出**末尾必须**为 `Overall Score: X.X/10`，用正则 `Overall Score:\s*(\d+\.?\d*)` 提取。
- `score ≥ 7.0` → `approved` + 面试邀请；否则 → `rejected` + 拒信。
- 结果通过 LinkedIn 回写候选人。

### 3.5 面试题生成（含人工审批）

- 3 类各若干题，Prompt 限定"技术 4 + 经验 3 + 问题解决 3"。
- 同样以 `NodeInterrupt` 挂起，`interview_questions_approved = True` 后方可结束流程（当前实现直接 `END`）。

------

## 四、启动安装说明

### 4.1 环境要求

- **Python ≥ 3.10**
- 可访问 `dashscope.aliyuncs.com`（通义）、`google.serper.dev`（Serper）、`www.linkedin.com`（Voyager）
- **有效的 LinkedIn 登录 Cookie**（见下方 4.4）

### 4.2 安装依赖

bash

```
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install \
  langgraph langchain-openai langchain-core langchain-community \
  pydantic python-dotenv requests ipython
```

### 4.3 配置环境变量

在项目根目录创建 `.env`：

ini

```
# 通义 / DashScope
DASHSCOPE_API_KEY=sk-xxxxxxxx
# 或（二选一，代码会回退）
TONGYI_API_KEY=sk-xxxxxxxx

# Google Serper（用于 LinkedIn 搜索）
SERPER_API_KEY=xxxxxxxx

# LinkedIn Voyager Cookie（从浏览器 DevTools 复制完整 cookie）
LINKEDIN_COOKIE=li_at=AQE...; JSESSIONID="ajax:xxx"; ...
```

> `load_dotenv()` 会自动加载。代码对 `DASHSCOPE_API_KEY` 与 `TONGYI_API_KEY` 做回退兼容。

### 4.4 获取 LinkedIn Cookie（关键步骤）

1. 浏览器登录 `linkedin.com`。
2. 打开 DevTools → Network → 任意 `voyager/api` 请求。
3. 复制请求头中的 `cookie` 字段整串（包含 `li_at`、`JSESSIONID` 等）。
4. 填入 `.env` 的 `LINKEDIN_COOKIE`。

> ⚠️ Cookie 会过期（通常数天到数周），失效后 `get_linkedin_profile` 与 `send_linkedin_message` 会返回 `{"error": ...}`。

### 4.5 运行

bash

```
python recruiter_agent.py
```

程序会：

1. 交互式采集 12 项岗位需求；
2. 生成 JD → 抛 `NodeInterrupt`（控制台打印 "Human Review Required"）；
3. 脚本中自动把 `job_description_approved` 设为 `True` 以演示续跑；
4. 依次执行 LinkedIn 触达、CV 分析、面试题生成（再次 `NodeInterrupt`）。

------

## 五、程序实践

### 5.1 交互输入示例

text

```
What is the job title? Senior Backend Engineer
Provide a description of the company: AI social publishing SaaS
List the job requirements (comma-separated): Python,FastAPI,PostgreSQL
List the candidate responsibilities (comma-separated): Design REST APIs,Own DB schema,Code review
List the candidate qualifications (comma-separated): 5+ years backend,CS degree
List the company benefits (comma-separated): Remote,Equity,Learning budget
What is the interview date (YYYY-MM-DD)? 2026-11-01
What is the preferred country for the role? China
What is the required years of experience? 5
Provide any LinkedIn profile URLs (comma-separated, optional): 
List the required skills (comma-separated): Python,Redis,Docker
What is the salary range for the role? 30k-50k CNY/month
```

### 5.2 运行态输出示例

text

```
New State Update:
Phase: requirements_gathering
Latest Message: Let's begin gathering the job requirements. What is the job title?

Human Review Required: Please review the generated job description:

# Senior Backend Engineer @ AI Social Publishing SaaS
## About the Company
...
## Role Overview
...
```

审批后续跑（demo 中已在脚本里自动执行）：

python

```
recruitment_workflow.update_state(config, {"job_description_approved": True})
```

### 5.3 三种典型使用姿势

| 姿势            | 做法                                                         | 适用场景                                              |
| --------------- | ------------------------------------------------------------ | ----------------------------------------------------- |
| **演示 / 学习** | 直接跑脚本（内置自动审批 + 自动续跑）                        | 了解 LangGraph `NodeInterrupt` 与 `update_state` 用法 |
| **半自动招聘**  | 保留脚本但把"自动审批"改成**真正的 CLI 询问**（`input("approve? y/n")`） | HR 审 JD / 面试题后再放行                             |
| **生产集成**    | 把 `input()` 换成 Web 表单 / Slack 卡片，`MemorySaver` 换 `SqliteSaver` | 团队协同、跨进程断点续跑                              |

------

## 六、目录结构建议

当前单文件版本建议拆分如下：

text

```
RecruiterAgent/
├── recruiter_agent.py         # 当前单文件版本
├── .env                        # 密钥（勿提交 Git）
├── requirements.txt
└── README.md
```

工程化拆分：

text

```
recruiter/
├── config.py                   # llm / search 单例、密钥加载
├── models.py                   # JobRequirements / CandidateProfile / RecruitmentState
├── integrations/
│   ├── serper.py               # search_linkedin_candidates
│   └── linkedin.py             # get_linkedin_profile / send_linkedin_message
├── nodes/
│   ├── requirements.py
│   ├── job_desc.py
│   ├── linkedin.py
│   ├── cv.py
│   └── interview.py
├── routing.py                  # route_after_job_desc / route_after_cv
└── graph.py                    # 装配与 compile
```