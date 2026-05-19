<p align="center">
  <img alt="Agentic RAG for Dummies Logo" src="assets/logo.png" width="350px">
</p>

<h1 align="center">Agentic RAG </h1>

<p align="center">
  <strong>使用 LangGraph、对话记忆和人工参与的查询澄清，构建模块化的 Agentic RAG 系统</strong>
</p>

## 概述

本仓库演示了如何使用 LangGraph 构建一个 **Agentic RAG（检索增强生成）** 系统。大多数 RAG 教程只展示基本概念，但缺乏构建模块化、智能体驱动系统的指导——本项目通过提供**可扩展架构**来弥补这一差距。

### 功能特性

| 特性 | 描述 |
|---|---|
| 🗂️ **层级索引** | 搜索小块以获得精确度，检索大块（Parent 块）以获取上下文 |
| 🧠 **对话记忆** | 跨问题维持上下文，实现自然对话 |
| ❓ **查询澄清** | 重写模糊查询，或暂停并向用户询问详细信息 |
| 🤖 **智能体编排** | LangGraph 协调完整的检索和推理工作流 |
| 🔀 **多智能体 Map-Reduce** | 将复杂查询分解为并行子查询 |
| ✅ **自我纠正** | 初始结果不足时自动重新查询 |
| 🗜️ **上下文压缩** | 在长检索循环中保持工作记忆的简洁 |
| 🔍 **可观测性** | 支持集成 Langfuse 跟踪 LLM 调用、工具使用和图执行 |

## 项目结构

模块化设计，每个组件都可独立替换。

```
project/
├── app.py                    # 主 Gradio 应用入口点
├── config.py                 # 配置中心（模型、块大小、提供商）
├── document_chunker.py       # PDF 文档分块工具
├── core/                     # RAG 系统编排
├── db/                       # 向量数据库和父块存储
├── rag_agent/                # LangGraph 工作流（节点、边、提示词、工具、状态）
└── ui/                       # Gradio 界面
```

## 工作原理

### 查询处理流程

```
用户查询 → 对话摘要 → 查询重写 → 查询澄清 →
并行智能体推理 → 聚合 → 最终响应
```

1. **对话理解** — 从对话历史中提取上下文
2. **查询澄清** — 重写模糊查询，必要时暂停等待人工输入
3. **智能检索** — 多智能体并行执行子查询，搜索子块 → 获取父块 → 自我纠正 → 上下文压缩
4. **响应生成** — 聚合所有智能体的结果为一个连贯答案

## LLM 提供商

本系统与提供商无关，一行代码即可切换。

| 提供商 | 安装 | 模型示例 |
|---|---|---|
| **Ollama（本地）** | `ollama pull qwen3:4b-instruct-2507-q4_K_M` | `ChatOllama(model="qwen3:4b-instruct-2507-q4_K_M")` |
| **OpenAI** | `pip install langchain-openai` | `ChatOpenAI(model="gpt-4o-mini")` |
| **Anthropic** | `pip install langchain-anthropic` | `ChatAnthropic(model="claude-sonnet-4-5-20250929")` |
| **Google Gemini** | `pip install langchain-google-genai` | `ChatGoogleGenerativeAI(model="gemini-2.5-flash")` |

> ⚠️ 本地模型建议使用 **7B+** 以确保工具调用可靠性。

## 快速开始

```bash
pip install -r requirements.txt
# 将 PDF 放入 docs/ 目录
python project/app.py
```

打开 `http://127.0.0.1:7860` 开始聊天。

### Docker

```bash
docker build -t agentic-rag -f project/Dockerfile .
docker run -p 7860:7860 agentic-rag
```

## 架构流程图

参见 [assets/agentic_rag_workflow.png](assets/agentic_rag_workflow.png)。