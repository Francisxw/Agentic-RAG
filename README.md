<p align="center">
  <img alt="Agentic RAG for Dummies Logo" src="assets/logo.png" width="350px">
</p>

<h1 align="center">Agentic RAG </h1>

<p align="center">
  <strong>使用 LangGraph、对话记忆和人工参与的查询澄清，构建模块化的 Agentic RAG 系统</strong>
</p>

---

## 概述

本仓库是一个使用 LangGraph 构建的 **Agentic RAG（检索增强生成）** 系统。它在传统 RAG "检索→生成"的基础上，引入**多智能体并行推理、对话记忆、查询澄清、自我纠正、上下文压缩**等能力，将 RAG 从被动检索升级为主动推理。

### 核心亮点

| 特性 | 说明 |
|---|---|
| **多智能体 Map-Reduce** | 复杂查询自动拆分为子问题，每个子问题由独立智能体并行处理，最后聚合结果 |
| **层级索引（Parent-Child Chunking）** | 精确搜索小块 + 大块上下文，兼顾召回率和信息完整性 |
| **人工参与的查询澄清** | LLM 检测到模糊查询时自动暂停，等待用户澄清后再继续，避免浪费检索资源 |
| **完整的 UI 支持** | 基于 Gradio 的 Web 界面，支持文档上传管理、流式输出、Chatbot 交互 |
| **多文档管理** | 支持 PDF 上传 → 自动转换 → 向量化索引的完整生命周期 |
| **可观测性集成** | 可选集成 Langfuse，追踪 LLM 调用、工具使用、图执行全链路 |
| **MinerU 高质量 PDF 解析** | 优先使用 MinerU 引擎进行版面分析 + OCR 解析，降级到 pymupdf4llm |
| **混合检索** | 稠密向量（语义相似度）+ 稀疏向量（BM25 关键词匹配）双重搜索 |

---

## 总体架构

### 系统分层

```
┌──────────────────────────────────────────────────────────┐
│                     UI Layer (Gradio)                      │
│    project/ui/gradio_app.py   project/core/chat_interface  │
├──────────────────────────────────────────────────────────┤
│                   Orchestration Layer (LangGraph)          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │ 对话摘要  │→ │ 查询重写  │→ │  智能体   │→ │ 答案聚合  │ │
│  │          │  │+ 澄清中断 │  │ (并行×N) │  │          │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │
│                     rag_agent/                             │
├──────────────────────────────────────────────────────────┤
│               Retrieval Layer (LangChain + Qdrant)         │
│   ┌─────────────┐  ┌─────────────┐  ┌───────────────┐   │
│   │ 稠密向量检索  │  │ 稀疏向量检索  │  │ 父块文件存储   │   │
│   │ (all-mpnet)  │  │  (BM25)     │  │ (JSON on disk)│   │
│   └─────────────┘  └─────────────┘  └───────────────┘   │
│                     db/                                    │
├──────────────────────────────────────────────────────────┤
│                 Document Processing Layer                   │
│    MinerU / pymupdf4llm → Markdown → Chunking (Parent+Child)│
│                     utils.py / document_chunker.py          │
├──────────────────────────────────────────────────────────┤
│                  Observability Layer (Optional)             │
│                    Langfuse Core/observability.py            │
└──────────────────────────────────────────────────────────┘
```

### 项目结构

```
project/
├── app.py                        # Gradio 应用入口
├── config.py                     # 集中配置（模型、分块、检索参数等）
├── document_chunker.py           # 父子层级分块器
├── utils.py                      # PDF→Markdown 转换、token 估算等工具
│
├── core/
│   ├── rag_system.py             # RAG 系统入口，组装所有组件
│   ├── chat_interface.py         # 流式聊天逻辑（处理工具调用、系统节点展示）
│   ├── document_manager.py       # 文档上传→转换→索引的完整生命周期
│   └── observability.py          # Langfuse 可观测性集成
│
├── db/
│   ├── vector_db_manager.py      # Qdrant 向量数据库管理（稠密+稀疏混合检索）
│   └── parent_store_manager.py   # 父块文件存储管理器
│
├── rag_agent/                    # 核心：LangGraph 智能体工作流
│   ├── graph.py                  # 图构建（主图 + 智能体子图两层结构）
│   ├── graph_state.py            # 状态定义（主图 State + 子图 AgentState）
│   ├── nodes.py                  # 所有图节点函数
│   ├── edges.py                  # 条件路由函数
│   ├── prompts.py                # 所有系统提示词
│   ├── schemas.py                # Pydantic 结构化输出模型
│   └── tools.py                  # 检索工具工厂（search_child_chunks, retrieve_parent_chunks）
│
├── ui/
│   ├── gradio_app.py             # Gradio UI 构建（文档管理 Tab + 聊天 Tab）
│   └── css.py                    # 自定义样式
│
└── assets/
    └── chatbot_avatar.png        # Chatbot 头像
```

---

## 详细运行流程

### 完整查询生命周期

```
用户输入
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  主图（Main Graph）                                    │
│                                                       │
│  1. summarize_history                                 │
│     │  读取对话历史 → 生成 1-2 句摘要                 │
│     │  同时重置 agent_answers（新查询周期开始）        │
│     ▼                                                 │
│  2. rewrite_query                                     │
│     │  融合对话摘要，调用 LLM 进行查询分析：           │
│     │  ├─ 如果清晰 → 拆分为多个自包含子查询            │
│     │  └─ 如果不清晰 → 返回澄清请求，中断等用户输入    │
│     ├──────────────────────┐                          │
│     ▼ (清晰)               ▼ (不清晰)                 │
│  3. 并行智能体            request_clarification       │
│     (Send API × N)         │  等待用户澄清后回到 2    │
│     │                       │                          │
│     ▼                       │                          │
│  4. 每个智能体子图         │                          │
│     (独立 Agent 实例)      │                          │
│     │                       │                          │
│     ▼                       │                          │
│  5. aggregate_answers       │                          │
│     │  所有智能体结果排序聚合                           │
│     ▼                       │                          │
│  END ───────────────────────┘                          │
│                                                       │
└─────────────────────────────────────────────────────┘
                                                   
输出给用户
```

### 智能体子图内部流程

这是系统最核心的部分。每个子查询都会启动一个独立的智能体实例：

```
agent_subgraph START
    │
    ▼
┌───────────────────────────────────────────────────────┐
│  orchestrator（编排器）                                  │
│  ├─ 判断是否有压缩上下文（来自之前的迭代）              │
│  ├─ 首次运行：强制调用 search_child_chunks              │
│  └─ 后续运行：基于已有信息决定继续检索还是生成答案      │
└──────────┬────────────────────────────────────────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
  有工具调用    无工具调用 (答案已完整)
     │           │
     ▼           ▼
┌──────────┐  collect_answer
│  tools   │  → 打包最终答案，附带索引
│ (工具节点)│
└────┬─────┘
     ▼
┌──────────────────────┐
│ should_compress_     │  ← 检查 token 是否超阈值
│ context（条件节点）     │
└──────┬───────┬──────┘
       ▼       ▼
   超阈值     未超阈值
       │       │
       ▼       │
 compress_     │
 context       │  ← 压缩后清除历史消息，只保留摘要
 (LLM 压缩)    │    避免重复检索
       │       │
       └──┬────┘
          ▼
 orchestrator（回到编排器，继续迭代直至答案完整）
          │
          ▼
   ... (循环) ...
          │
   工具调用超限 → fallback_response → collect_answer
   迭代超限    → fallback_response → collect_answer
   答案完整    → collect_answer
                    │
                    ▼
              agent_subgraph END
```

### 文档处理流程

```
用户上传 PDF
    │
    ▼
┌─────────────────────────────┐
│  MinerU 解析（首选）          │
│  版面分析 + OCR + VLM        │
│  ↓ 失败则降级                 │
│  pymupdf4llm 纯文本提取       │
└──────────┬──────────────────┘
           ▼
    Markdown 文件
           │
           ▼
┌─────────────────────────────┐
│  层级分块器（document_chunker）│
│                              │
│  1. 按 H1/H2/H3 标题分割     │
│     → 初步父块               │
│  2. 合并过小的父块           │
│     ( < MIN_PARENT_SIZE )    │
│  3. 拆分过大的父块           │
│     ( > MAX_PARENT_SIZE )    │
│  4. 清理残余小块             │
│  5. 每个父块 → 子块          │
│     (CHILD_CHUNK_SIZE=500)   │
└──────────┬──────────────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
  子块存入      父块保存
  Qdrant 向量库  JSON 文件
  (稠密+稀疏    (文件系统)
   双向量)
```

### 检索路径图

```
search_child_chunks(query, k=5-7)
    │
    ┌── 未找到相关内容 → 重新表述查询再搜索
    │
    ▼ 找到相关子块
    返回：parent_id + 文件名 + 片段内容
    │
    ▼
retrieve_parent_chunks(parent_id)   ← 每个 parent_id 逐个调用
    │
    ▼
    返回：完整父块内容（大上下文）
    │
    ▼
    编排器基于完整上下文生成答案
```

---

## 关键技术细节

### 1. 两层图架构

系统使用 LangGraph 的 **两层嵌套图** 架构：

- **主图（Main Graph）**：负责对话级编排——摘要、重写、分发、聚合。使用 `MessagesState` 管理对话历史。
- **智能体子图（Agent Subgraph）**：负责单个查询的检索推理循环。使用 `AgentState` 管理工具调用计数、上下文压缩等。

这种分离使得：
- 主图可以**并行**启动多个子图（通过 `Send` API）
- 每个子图有独立的递归限制和工具调用预算
- 子图的内部状态不会污染主图

### 2. 状态设计

```python
# 主图状态
class State(MessagesState):
    questionIsClear: bool                # 查询是否清晰
    conversation_summary: str            # 对话摘要
    originalQuery: str                   # 原始用户查询
    rewrittenQuestions: List[str]        # 重写后的子问题列表
    agent_answers: List[dict]            # 所有智能体的答案

# 智能体子图状态
class AgentState(MessagesState):
    question: str                        # 当前智能体负责的子问题
    question_index: int                  # 问题索引
    tool_call_count: int                 # 工具调用累计（达到 MAX_TOOL_CALLS=8 触发回退）
    iteration_count: int                 # 迭代次数累计（达到 MAX_ITERATIONS=10 触发回退）
    context_summary: str                 # 压缩后的上下文
    retrieval_keys: Set[str]             # 已检索过的 key（避免重复检索）
    final_answer: str                    # 最终答案
```

关键设计：`tool_call_count` 和 `iteration_count` 使用 `operator.add` 实现累加，`retrieval_keys` 使用自定义 `set_union` 实现去重合并。

### 3. 上下文压缩机制

智能体在检索循环中，消息列表会不断增长（工具输入 + 工具输出）。当 token 数超过阈值时，触发压缩：

1. `should_compress_context` 估算当前 token 总量
2. 若超过 `BASE_TOKEN_THRESHOLD (2000) + 已有摘要 * 0.9`，进入压缩
3. `compress_context` 调用 LLM 将检索对话压缩为结构化摘要
4. 删除旧消息，仅保留系统消息和摘要
5. 注入已检索的 key 列表，避免编排器重复检索

### 4. 查询澄清中断机制

```
rewrite_query → is_clear=False
    → 图在 request_clarification 前中断（interrupt_before）
    → 用户输入澄清
    → rewrite_query 再次执行（携带澄清 + 原查询上下文）
```

这利用了 LangGraph 的 `interrupt_before` + checkpointer 功能，实现**图执行中的暂停与恢复**。

### 5. 混合检索

向量搜索使用 **Dense + Sparse 双向量**：

| 向量类型 | 模型 | 作用 |
|---|---|---|
| 稠密向量 (Dense) | `all-mpnet-base-v2` (768维) | 捕捉语义相似度 |
| 稀疏向量 (Sparse) | `Qdrant/bm25` | 捕捉关键词匹配 |

检索模式为 `RetrievalMode.HYBRID`，综合两种向量的分数排序。

### 6. 系统提示词设计

系统使用 6 组专门的系统提示词，每个图节点各有分工：

| 提示词 | 使用节点 | 用途 |
|---|---|---|
| `get_conversation_summary_prompt` | summarize_history | 对话摘帽 → 摘要 |
| `get_rewrite_query_prompt` | rewrite_query | 查询重写 + 清晰度判断 |
| `get_orchestrator_prompt` | orchestrator | 检索策略 + 答案生成 |
| `get_fallback_response_prompt` | fallback_response | 预算耗尽时的降级生成 |
| `get_context_compression_prompt` | compress_context | 检索日志 → 结构化摘要 |
| `get_aggregation_prompt` | aggregate_answers | 多智能体结果 → 统一答案 |

---

## LLM 提供商

系统与提供商无关，一行代码切换。

| 提供商 | 安装 | 配置示例 |
|---|---|---|
| **Ollama（本地）** | `ollama pull qwen3:8b` | `ChatOllama(model="qwen3:8b")` |
| **OpenAI** | `pip install langchain-openai` | `ChatOpenAI(model="gpt-4o-mini")` |
| **Anthropic** | `pip install langchain-anthropic` | `ChatAnthropic(model="claude-sonnet-4-5-20250929")` |
| **Google Gemini** | `pip install langchain-google-genai` | `ChatGoogleGenerativeAI(model="gemini-2.5-flash")` |

> 本地模型建议 7B+ 以确保工具调用可靠性。Ollama 模型在 `project/config.py` 的 `LLM_MODEL` 中配置。

---

## 快速开始

```bash
pip install -r requirements.txt
# 将 PDF 放入 docs/ 目录，或通过 UI 上传
python project/app.py
```

打开 `http://127.0.0.1:7860`，在 Documents 页面上传文档，然后在 Chat 页面开始提问。

### Docker

```bash
docker build -t agentic-rag -f project/Dockerfile .
docker run -p 7860:7860 agentic-rag
```

---

## 配置文件

核心参数位于 `project/config.py`，可根据需求调整：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `LLM_MODEL` | `qwen3:8b` | LLM 模型名称 |
| `LLM_TEMPERATURE` | `0` | LLM 温度（事实性任务建议 0） |
| `DENSE_MODEL` | `all-mpnet-base-v2` | 稠密嵌入模型 |
| `SPARSE_MODEL` | `Qdrant/bm25` | 稀疏嵌入模型 |
| `CHILD_CHUNK_SIZE` | `500` | 子块大小（字符数） |
| `MAX_PARENT_SIZE` | `4000` | 父块最大字符数 |
| `MAX_TOOL_CALLS` | `8` | 智能体最大工具调用次数 |
| `MAX_ITERATIONS` | `10` | 智能体最大迭代次数 |
| `BASE_TOKEN_THRESHOLD` | `2000` | 上下文压缩触发阈值 |
| `LANGFUSE_ENABLED` | `false` | 是否开启 Langfuse 跟踪 |

---

## 架构流程图

详见 [assets/agentic_rag_workflow.png](assets/agentic_rag_workflow.png)。

## 许可证

MIT