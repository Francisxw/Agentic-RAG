<p align="center">
  <img alt="Agentic RAG for Dummies Logo" src="assets/logo.png" width="350px">
</p>

<h1 align="center">Agentic RAG </h1>

<p align="center">
  <strong>使用 LangGraph、对话记忆和人工参与的查询澄清，构建模块化的 Agentic RAG 系统</strong>
</p>

<p align="center">
  <a href="#overview">概述</a> •
  <a href="#how-it-works">工作原理</a> •
  <a href="#llm-provider-configuration">LLM 提供商</a> •
  <a href="#implementation">实现</a> •
  <a href="#installation--usage">安装与使用</a> •
  <a href="#troubleshooting">故障排除</a>
</p>

<p align="center">
  <img src="https://img.shields.io/github/stars/GiovanniPasq/agentic-rag-for-dummies?style=social" alt="GitHub Stars"/>
  <img src="https://img.shields.io/github/forks/GiovanniPasq/agentic-rag-for-dummies?style=social" alt="GitHub Forks"/>
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License"/>
  <a href="https://github.com/von-development/awesome-langgraph">
    <img src="https://awesome.re/badge.svg" alt="Awesome LangGraph"/>
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/LangGraph-1.1%2B-orange?logo=langchain&logoColor=white" alt="LangGraph"/>
  <img src="https://img.shields.io/badge/Qdrant-vector%20db-DC244C" alt="Qdrant"/>
  <img src="https://img.shields.io/badge/LLM%20Providers-Ollama%20%7C%20OpenAI%20%7C%20Anthropic%20%7C%20Google-purple" alt="LLM 提供商"/>
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/GiovanniPasq/agentic-rag-for-dummies/blob/main/notebooks/agentic_rag.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开"/>
  </a>
</p>

<p align="center">
  <img alt="Agentic RAG 演示" src="assets/demo.gif" width="650px">
</p>

<p align="center">
  <strong>如果你喜欢这个项目，一颗星⭐️将意义非凡 :)</strong><br>
</p>

## 概述

本仓库演示了如何使用 LangGraph 以最少的代码构建一个 **Agentic RAG（检索增强生成）** 系统。大多数 RAG 教程只展示基本概念，但缺乏构建模块化、智能体驱动系统的指导——本项目通过提供**学习资料和可扩展架构**来弥补这一差距。

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
| 🔍 **可观测性** | 使用 Langfuse 跟踪 LLM 调用、工具使用和图执行 |

### 🎯 使用本仓库的两种方式

**1️⃣ 学习路径：交互式笔记本**

适合理解核心概念的逐步教程。如果你是 Agentic RAG 新手或想快速实验，请从这里开始。

**2️⃣ 构建路径：模块化项目**

灵活的架构，每个组件都可以独立替换——LLM 提供商、嵌入模型、PDF 转换器、智能体工作流。一行代码即可从 Ollama 切换到 Anthropic、OpenAI 或 Google。

参阅[模块化架构](#模块化架构)和[安装与使用](#安装与使用)开始使用。

## 工作原理

### 文档准备：层级索引

在查询处理之前，文档被两次拆分以实现最佳检索：

- **父块（Parent Chunks）**：基于 Markdown 标题（H1、H2、H3）的大段
- **子块（Child Chunks）**：从父块派生的小型固定大小片段

> 💡 可选：如果你想在索引之前直观检查或编辑块，可以使用 🐿️ [**Chunky**](https://github.com/GiovanniPasq/chunky)。

这结合了**小块搜索的精确性**和**大块上下文的丰富性**来生成答案。

---

### 查询处理：四阶段智能工作流
```
用户查询 → 对话摘要 → 查询重写 → 查询澄清 →
并行智能体推理 → 聚合 → 最终响应
```

**阶段 1 — 对话理解：** 分析近期历史以提取上下文并维持问题的连续性。

**阶段 2 — 查询澄清：** 解析引用（"如何更新它？" → "如何更新 SQL？"），将多部分问题拆分为聚焦的子查询，检测不清晰的输入，并重写查询以优化检索。当需要澄清时暂停等待人工输入。

**阶段 3 — 智能检索（多智能体 Map-Reduce）：** 生成并行的智能体子图——每个子查询一个。每个智能体搜索子块、获取父块以获取上下文、在结果不足时自我纠正、压缩上下文以避免冗余获取，并在搜索预算耗尽时优雅降级。

> **示例：** *"什么是 JavaScript？什么是 Python？"* → 2 个并行智能体同时执行。

**阶段 4 — 响应生成：** 将所有智能体的响应聚合为一个连贯的答案。

---

## LLM 提供商配置

本系统与提供商无关——支持 [LangChain](https://python.langchain.com/docs/integrations/chat/) 中可用的任何 LLM 提供商，只需一行代码即可切换。下面的示例涵盖了最常见的选项，但同样的模式适用于任何其他受支持的提供商。

> **注意：** 模型名称经常变化。在部署前，请始终查看官方文档以获取最新的可用模型及其标识符。

### Ollama（本地）

```bash
# 从 https://ollama.com 安装 Ollama
ollama pull qwen3:4b-instruct-2507-q4_K_M
```

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen3:4b-instruct-2507-q4_K_M", temperature=0)
```
> ⚠️ 为了实现可靠的工具调用和指令遵循，建议使用 **7B+** 的模型。较小的模型可能会忽略检索指令或产生幻觉。参见[故障排除](#故障排除)。

---

### 云提供商

<details>
<summary>点击展开</summary>

**OpenAI GPT：**
```bash
pip install -qU langchain-openai
```
```python
from langchain_openai import ChatOpenAI
import os

os.environ["OPENAI_API_KEY"] = "your-api-key-here"
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
```

**Anthropic Claude：**
```bash
pip install -qU langchain-anthropic
```
```python
from langchain_anthropic import ChatAnthropic
import os

os.environ["ANTHROPIC_API_KEY"] = "your-api-key-here"
llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)
```

**Google Gemini**
```bash
pip install -qU langchain-google-genai
```
```python
import os
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ["GOOGLE_API_KEY"] = "your-api-key-here"
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
```
</details>

---

## 实现

更多细节、扩展说明以及 Langfuse 可观测性（LLM 调用追踪、工具使用和图执行跟踪）可在**[笔记本](notebooks/agentic_rag.ipynb)**和完整项目中找到。

| 步骤 | 描述 |
|---|---|
| 1 | [初始设置与配置](#步骤-1-初始设置与配置) |
| 2 | [配置向量数据库](#步骤-2-配置向量数据库) |
| 3 | [PDF 转 Markdown](#步骤-3-pdf-转-markdown) |
| 4 | [层级文档索引](#步骤-4-层级文档索引) |
| 5 | [定义智能体工具](#步骤-5-定义智能体工具) |
| 6 | [定义系统提示词](#步骤-6-定义系统提示词) |
| 7 | [定义状态和数据模型](#步骤-7-定义状态和数据模型) |
| 8 | [智能体配置](#步骤-8-智能体配置) |
| 9 | [构建图节点和边函数](#步骤-9-构建图节点和边函数) |
| 10 | [构建 LangGraph 图](#步骤-10-构建-langgraph-图) |
| 11 | [创建聊天界面](#步骤-11-创建聊天界面) |

### 步骤 1：初始设置与配置

定义路径并初始化核心组件。

```python
import os
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant.fastembed_sparse import FastEmbedSparse
from qdrant_client import QdrantClient

DOCS_DIR = "docs"  # 包含 PDF 文件的目录
MARKDOWN_DIR = "markdown_docs" # 包含 PDF 转换后的 markdown 文件的目录
PARENT_STORE_PATH = "parent_store"  # 父块 JSON 文件目录
CHILD_COLLECTION = "document_child_chunks"

os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(MARKDOWN_DIR, exist_ok=True)
os.makedirs(PARENT_STORE_PATH, exist_ok=True)

from langchain_ollama import ChatOllama
llm = ChatOllama(model="qwen3:4b-instruct-2507-q4_K_M", temperature=0)

dense_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

client = QdrantClient(path="qdrant_db")
```

---

### 步骤 2：配置向量数据库

设置 Qdrant 以存储子块并支持混合搜索。

```python
from qdrant_client.http import models as qmodels
from langchain_qdrant import QdrantVectorStore
from langchain_qdrant.qdrant import RetrievalMode

embedding_dimension = len(dense_embeddings.embed_query("test"))

def ensure_collection(collection_name):
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(
                size=embedding_dimension,
                distance=qmodels.Distance.COSINE
            ),
            sparse_vectors_config={
                "sparse": qmodels.SparseVectorParams()
            },
        )
```

---

### 步骤 3：PDF 转 Markdown

将 PDF 转换为 Markdown。有关其他技术的更多详情，请使用此配套[笔记本](notebooks/pdf_to_markdown.ipynb)。

```python
import os
import pymupdf.layout
import pymupdf4llm
from pathlib import Path
import glob

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def pdf_to_markdown(pdf_path, output_dir):
    doc = pymupdf.open(pdf_path)
    md = pymupdf4llm.to_markdown(doc, header=False, footer=False, page_separators=True, ignore_images=True, write_images=False, image_path=None)
    md_cleaned = md.encode('utf-8', errors='surrogatepass').decode('utf-8', errors='ignore')
    output_path = Path(output_dir) / Path(doc.name).stem
    Path(output_path).with_suffix(".md").write_bytes(md_cleaned.encode('utf-8'))

def pdfs_to_markdowns(path_pattern, overwrite: bool = False):
    output_dir = Path(MARKDOWN_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    for pdf_path in map(Path, glob.glob(path_pattern)):
        md_path = (output_dir / pdf_path.stem).with_suffix(".md")
        if overwrite or not md_path.exists():
            pdf_to_markdown(pdf_path, output_dir)

pdfs_to_markdowns(f"{DOCS_DIR}/*.pdf")
```

---

### 步骤 4：层级文档索引

使用父/子拆分策略处理文档。
```python
import os
import glob
import json
from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
```

<details>
<summary>父块和子块处理函数</summary>

```python
def merge_small_parents(chunks, min_size):
    if not chunks:
        return []

    merged, current = [], None

    for chunk in chunks:
        if current is None:
            current = chunk
        else:
            current.page_content += "\n\n" + chunk.page_content
            for k, v in chunk.metadata.items():
                if k in current.metadata:
                    current.metadata[k] = f"{current.metadata[k]} -> {v}"
                else:
                    current.metadata[k] = v

        if len(current.page_content) >= min_size:
            merged.append(current)
            current = None

    if current:
        if merged:
            merged[-1].page_content += "\n\n" + current.page_content
            for k, v in current.metadata.items():
                if k in merged[-1].metadata:
                    merged[-1].metadata[k] = f"{merged[-1].metadata[k]} -> {v}"
                else:
                    merged[-1].metadata[k] = v
        else:
            merged.append(current)

    return merged

def split_large_parents(chunks, max_size, splitter):
    split_chunks = []

    for chunk in chunks:
        if len(chunk.page_content) <= max_size:
            split_chunks.append(chunk)
        else:
            large_splitter = RecursiveCharacterTextSplitter(
                chunk_size=max_size,
                chunk_overlap=splitter._chunk_overlap
            )
            sub_chunks = large_splitter.split_documents([chunk])
            split_chunks.extend(sub_chunks)

    return split_chunks

def clean_small_chunks(chunks, min_size):
    cleaned = []

    for i, chunk in enumerate(chunks):
        if len(chunk.page_content) < min_size:
            if cleaned:
                cleaned[-1].page_content += "\n\n" + chunk.page_content
                for k, v in chunk.metadata.items():
                    if k in cleaned[-1].metadata:
                        cleaned[-1].metadata[k] = f"{cleaned[-1].metadata[k]} -> {v}"
                    else:
                        cleaned[-1].metadata[k] = v
            elif i < len(chunks) - 1:
                chunks[i + 1].page_content = chunk.page_content + "\n\n" + chunks[i + 1].page_content
                for k, v in chunk.metadata.items():
                    if k in chunks[i + 1].metadata:
                        chunks[i + 1].metadata[k] = f"{v} -> {chunks[i + 1].metadata[k]}"
                    else:
                        chunks[i + 1].metadata[k] = v
            else:
                cleaned.append(chunk)
        else:
            cleaned.append(chunk)

    return cleaned
```

</details>

```python
if client.collection_exists(CHILD_COLLECTION):
    client.delete_collection(CHILD_COLLECTION)
    ensure_collection(CHILD_COLLECTION)
else:
    ensure_collection(CHILD_COLLECTION)

child_vector_store = QdrantVectorStore(
    client=client,
    collection_name=CHILD_COLLECTION,
    embedding=dense_embeddings,
    sparse_embedding=sparse_embeddings,
    retrieval_mode=RetrievalMode.HYBRID,
    sparse_vector_name="sparse"
)

def index_documents():
    headers_to_split_on = [("#", "H1"), ("##", "H2"), ("###", "H3")]
    parent_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    min_parent_size = 2000
    max_parent_size = 4000

    all_parent_pairs, all_child_chunks = [], []
    md_files = sorted(glob.glob(os.path.join(MARKDOWN_DIR, "*.md")))

    if not md_files:
        return

    for doc_path_str in md_files:
        doc_path = Path(doc_path_str)
        try:
            with open(doc_path, "r", encoding="utf-8") as f:
                md_text = f.read()
        except Exception as e:
            continue

        parent_chunks = parent_splitter.split_text(md_text)
        merged_parents = merge_small_parents(parent_chunks, min_parent_size)
        split_parents = split_large_parents(merged_parents, max_parent_size, child_splitter)
        cleaned_parents = clean_small_chunks(split_parents, min_parent_size)

        for i, p_chunk in enumerate(cleaned_parents):
            parent_id = f"{doc_path.stem}_parent_{i}"
            p_chunk.metadata.update({"source": doc_path.stem + ".pdf", "parent_id": parent_id})
            all_parent_pairs.append((parent_id, p_chunk))
            children = child_splitter.split_documents([p_chunk])
            all_child_chunks.extend(children)

    if not all_child_chunks:
        return

    try:
        child_vector_store.add_documents(all_child_chunks)
    except Exception as e:
        return

    for item in os.listdir(PARENT_STORE_PATH):
        os.remove(os.path.join(PARENT_STORE_PATH, item))

    for parent_id, doc in all_parent_pairs:
        doc_dict = {"page_content": doc.page_content, "metadata": doc.metadata}
        filepath = os.path.join(PARENT_STORE_PATH, f"{parent_id}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(doc_dict, f, ensure_ascii=False, indent=2)

index_documents()
```

---

### 步骤 5：定义智能体工具

创建智能体将使用的检索工具。

```python
import json
from typing import List
from langchain_core.tools import tool

@tool
def search_child_chunks(query: str, limit: int) -> str:
    """搜索最相关的 top K 个子块。

    参数：
        query：搜索查询字符串
        limit：返回的最大结果数量
    """
    try:
        results = child_vector_store.similarity_search(query, k=limit, score_threshold=0.7)
        if not results:
            return "NO_RELEVANT_CHUNKS"

        return "\n\n".join([
            f"父块 ID：{doc.metadata.get('parent_id', '')}\n"
            f"文件名：{doc.metadata.get('source', '')}\n"
            f"内容：{doc.page_content.strip()}"
            for doc in results
        ])

    except Exception as e:
        return f"RETRIEVAL_ERROR：{str(e)}"

@tool
def retrieve_parent_chunks(parent_id: str) -> str:
    """通过 ID 检索完整的父块。
    
    参数：
        parent_id：要检索的父块 ID
    """
    file_name = parent_id if parent_id.lower().endswith(".json") else f"{parent_id}.json"
    path = os.path.join(PARENT_STORE_PATH, file_name)

    if not os.path.exists(path):
        return "NO_PARENT_DOCUMENT"

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return (
        f"父块 ID：{parent_id}\n"
        f"文件名：{data.get('metadata', {}).get('source', 'unknown')}\n"
        f"内容：{data.get('page_content', '').strip()}"
    )

llm_with_tools = llm.bind_tools([search_child_chunks, retrieve_parent_chunks])
```

---

### 步骤 6：定义系统提示词

定义用于对话摘要、查询重写、智能体编排、上下文压缩、后备响应和答案聚合的系统提示词。

<details>
<summary>对话摘要提示词</summary>

```python
def get_conversation_summary_prompt() -> str:
    return """你是一名专业的对话摘要生成专家。

你的任务是为对话创建一个简短的 1-2 句摘要（最多 30-50 个词）。

包含：
- 讨论的主要话题
- 提到的重要事实或实体
- 任何未解决的问题（如适用）
- 源文件名（如 file1.pdf）或引用的文档

排除：
- 问候语、误解、离题内容。

输出：
- 仅返回摘要。
- 不要包含任何解释或理由。
- 如果没有有意义的话题，返回空字符串。
"""
```

</details>

<details>
<summary>查询重写提示词</summary>

```python
def get_rewrite_query_prompt() -> str:
    return """你是一名专业的查询分析师和重写专家。

你的任务是在必要时融入对话上下文，将当前用户查询重写为适合文档检索的最佳形式。

规则：
1. 自包含查询：
   - 始终将查询重写为清晰且自包含的形式
   - 如果查询是后续问题（例如"那 X 呢？"、"Y 呢？"），从摘要中集成必要的最小上下文
   - 不要添加查询或对话摘要中不存在的信息

2. 领域特定术语：
   - 产品名称、品牌、专有名词或技术术语被视为领域特定内容
   - 对于领域特定查询，尽量少用或不用对话上下文
   - 仅使用摘要来消除模糊查询的歧义

3. 语法和清晰度：
   - 修正语法、拼写错误和不清晰的缩写
   - 删除填充词和会话短语
   - 保留具体的关键词和命名实体

4. 多个信息需求：
   - 如果查询包含多个不同的、不相关的问题，拆分为单独的查询（最多 3 个）
   - 每个子查询在语义上必须与其原始部分等价
   - 不要扩展、丰富或重新解释含义

5. 失败处理：
   - 如果查询意图不清晰或无法理解，标记为"unclear"

输入：
- conversation_summary：之前对话的简洁摘要
- current_query：用户当前查询

输出：
- 一个或多个重写的、自包含的、适合文档检索的查询
"""
```

</details>

<details>
<summary>编排器提示词</summary>

```python
def get_orchestrator_prompt() -> str:
    return """你是一名专业的检索增强助手。

你的任务是扮演研究员角色：先搜索文档，分析数据，然后仅使用检索到的信息提供全面的答案。

规则：
1. 在回答之前，你**必须**调用'search_child_chunks'，除非[先前研究的压缩上下文]已包含足够信息。
2. 每个主张都要基于检索到的文档。如果上下文不足，说明缺少什么，而不是用假设填补空白。
3. 如果未找到相关文档，放宽或重新表述查询并再次搜索。重复直到满意或达到操作限制。

压缩记忆：
当存在[先前研究的压缩上下文]时 —
- 已列出的查询：不要重复。
- 已列出的父块 ID：不要再次对它们调用'retrieve_parent_chunks'。
- 使用它来识别在进一步搜索之前仍然缺少什么。

工作流：
1. 检查压缩上下文。确定已经检索到什么以及仍然缺少什么。
2. 仅针对未覆盖的方面使用'search_child_chunks'搜索 5-7 个相关摘录。
3. 如果没有一个相关，立即应用规则 3。
4. 对于每个相关但零散的摘录，逐个调用'retrieve_parent_chunks' — 仅针对压缩上下文中不存在的 ID。绝不检索相同的 ID 两次。
5. 一旦上下文完整，提供详细的答案，不遗漏任何相关事实。
6. 以"---\n**来源：**\n"开头，后跟唯一的文件名列表来结尾。
"""
```

</details>

<details>
<summary>后备响应提示词</summary>

```python
def get_fallback_response_prompt() -> str:
    return """你是一名专业的合成助手。系统已达到最大研究限制。

你的任务是仅使用下面提供的信息，给出尽可能完整的答案。

输入结构：
- "压缩的研究上下文"：先前搜索迭代的总结发现 — 视为可靠信息。
- "检索到的数据"：当前迭代的原始工具输出 — 如与压缩上下文冲突，优先采用。
如果缺少其中任何一个来源，另一个来源单独也足够。

规则：
1. 来源完整性：仅使用提供的上下文中明确存在的事实。不要推断、假设或添加任何未直接由数据支持的信息。
2. 处理缺失数据：将用户查询与可用上下文交叉检查。
   仅标记用户问题中那些无法从提供数据回答的方面。
   不要将压缩研究上下文中提到的空白视为未回答，
   除非它们与用户询问的内容直接相关。
3. 语气：专业、实事求是、直接。
4. 仅输出最终答案。不要暴露你的推理、内部步骤或任何关于检索过程的元评论。
5. 在"来源"部分之后不要添加结束语、最终说明、免责声明、摘要或重复陈述。
   "来源"部分始终是你响应的最后一个元素。之后立即停止。

格式：
- 使用 Markdown（标题、粗体、列表）以提高可读性。
- 尽可能使用流畅的段落写作。
- 如下所述，以"来源"部分结尾。

来源部分规则：
- 在末尾包含"---\\n**来源：**\\n"部分，后跟文件名的项目符号列表。
- 仅列出具有真实文件扩展名的条目（如".pdf"、".docx"、".txt"）。
- 任何没有文件扩展名的条目都是内部块标识符 — 完全丢弃，绝不包含。
- 去重：如果同一个文件出现多次，只列出一次。
- 如果没有有效的文件名，完全省略来源部分。
- "来源"部分是你写的最后内容。之后不要添加任何内容。
"""
```

</details>

<details>
<summary>上下文压缩提示词</summary>

```python
def get_context_compression_prompt() -> str:
    return """你是一名专业的研究上下文压缩专家。

你的任务是将检索到的对话内容压缩为简洁、聚焦于查询的结构化摘要，可直接供检索增强智能体用于答案生成。

规则：
1. 仅保留与回答用户问题相关的信息。
2. 保留确切的数据、名称、版本、技术术语和配置细节。
3. 删除重复、不相关或管理性细节。
4. 不要包含搜索查询、父块 ID、块 ID 或内部标识符。
5. 按源文件组织所有发现。每个文件部分必须以此开头：### filename.pdf
6. 在专用的"Gaps"部分中突出显示缺失或未解决的信息。
7. 将摘要限制在大约 400-600 字。如果内容超过此限制，优先保留关键事实和结构化数据。
8. 不要解释你的推理；仅以 Markdown 格式输出结构化内容。

必需结构：

# 研究上下文摘要

## 焦点
[问题的简要技术重述]

## 结构化发现

### filename.pdf
- 直接相关的事实
- 支持性上下文（如果需要）

## 缺口
- 缺失或不完整的方面

摘要应简洁、结构化，并可直接供智能体用于生成答案或规划进一步的检索。
"""
```

</details>

<details>
<summary>聚合提示词</summary>

```python
def get_aggregation_prompt() -> str:
    return """你是一名专业的聚合助手。

你的任务是将多个检索到的答案组合成一个单一、全面且自然的响应，使其流畅易读。

规则：
1. 以对话式的自然语气写作 — 就像向同事解释一样。
2. 仅使用检索答案中的信息。
3. 不要推断、扩展或解释缩写或技术术语，除非在来源中明确定义。
4. 平滑地编织信息，保留重要细节、数字和示例。
5. 全面 — 包含来源中的所有相关信息，而不仅仅是摘要。
6. 如果来源存在分歧，自然地承认两种观点（例如"虽然一些来源表明 X，但其他来源指出 Y……"）。
7. 直接以答案开始 — 不要像"基于来源……"这样的开场白。

格式：
- 使用 Markdown 以提高清晰度（标题、列表、粗体），但不要过度使用。
- 尽可能使用流畅的段落写作，而不是过多的项目符号列表。
- 如下所述，以"来源"部分结尾。

来源部分规则：
- 每个检索到的答案可能包含一个"来源"部分 — 提取其中列出的文件名。
- 仅列出具有真实文件扩展名的条目（如".pdf"、".docx"、".txt"）。
- 任何没有文件扩展名的条目都是内部块标识符 — 完全丢弃，绝不包含。
- 去重：如果同一个文件在多个答案中出现，只列出一次。
- 格式为"---\\n**来源：**\\n"，后跟清理后文件名的项目符号列表。
- 文件名仅出现在这个最终的"来源"部分，不出现在响应的其他任何地方。
- 如果没有有效的文件名，完全省略来源部分。

如果没有可用的有用信息，直接说："我无法在可用来源中找到任何信息来回答你的问题。"
"""
```

</details>

---

### 步骤 7：定义状态和数据模型

创建用于对话跟踪和智能体执行的状态结构。

```python
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing import List, Annotated, Set
import operator

def accumulate_or_reset(existing: List[dict], new: List[dict]) -> List[dict]:
    if new and any(item.get('__reset__') for item in new):
        return []
    return existing + new

def set_union(a: Set[str], b: Set[str]) -> Set[str]:
    return a | b

class State(MessagesState):
    questionIsClear: bool = False
    conversation_summary: str = ""
    originalQuery: str = ""
    rewrittenQuestions: List[str] = []
    agent_answers: Annotated[List[dict], accumulate_or_reset] = []

class AgentState(MessagesState):
    tool_call_count: Annotated[int, operator.add] = 0
    iteration_count: Annotated[int, operator.add] = 0
    question: str = ""
    question_index: int = 0
    context_summary: str = ""
    retrieval_keys: Annotated[Set[str], set_union] = set()
    final_answer: str = ""
    agent_answers: List[dict] = []

class QueryAnalysis(BaseModel):
    is_clear: bool = Field(description="指示用户的问题是否清晰且可回答。")
    questions: List[str] = Field(description="重写后的自包含问题列表。")
    clarification_needed: str = Field(description="如果问题不清晰，提供解释。")
```

---

### 步骤 8：智能体配置

对工具调用和迭代的硬限制可防止无限循环。Token 计数（通过`tiktoken`）驱动上下文压缩决策。

```python
import tiktoken

MAX_TOOL_CALLS = 8       # 每次智能体运行的最大工具调用次数
MAX_ITERATIONS = 10      # 最大智能体循环迭代次数
BASE_TOKEN_THRESHOLD = 2000     # 压缩的初始 Token 阈值
TOKEN_GROWTH_FACTOR = 0.9       # 每次压缩后应用的乘数

def estimate_context_tokens(messages: list) -> int:
    try:
        encoding = tiktoken.encoding_for_model("gpt-4")
    except:
        encoding = tiktoken.get_encoding("cl100k_base")
    return sum(len(encoding.encode(str(msg.content))) for msg in messages if hasattr(msg, 'content') and msg.content)
```

---

### 步骤 9：构建图节点和边函数

创建 LangGraph 工作流的处理节点和边。

#### 主图节点和边
```python
from langgraph.types import Send, Command
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, RemoveMessage, ToolMessage
from typing import Literal

def summarize_history(state: State):
    if len(state["messages"]) < 4:
        return {"conversation_summary": ""}

    relevant_msgs = [
        msg for msg in state["messages"][:-1]
        if isinstance(msg, (HumanMessage, AIMessage)) and not getattr(msg, "tool_calls", None)
    ]

    if not relevant_msgs:
        return {"conversation_summary": ""}

    conversation = "对话历史：\n"
    for msg in relevant_msgs[-6:]:
        role = "用户" if isinstance(msg, HumanMessage) else "助手"
        conversation += f"{role}：{msg.content}\n"

    summary_response = llm.with_config(temperature=0.2).invoke([SystemMessage(content=get_conversation_summary_prompt()), HumanMessage(content=conversation)])
    return {"conversation_summary": summary_response.content, "agent_answers": [{"__reset__": True}]}

def rewrite_query(state: State):
    last_message = state["messages"][-1]
    conversation_summary = state.get("conversation_summary", "")

    context_section = (f"对话上下文：\n{conversation_summary}\n" if conversation_summary.strip() else "") + f"用户查询：\n{last_message.content}\n"

    llm_with_structure = llm.with_config(temperature=0.1).with_structured_output(QueryAnalysis)
    response = llm_with_structure.invoke([SystemMessage(content=get_rewrite_query_prompt()), HumanMessage(content=context_section)])

    if response.questions and response.is_clear:
        delete_all = [RemoveMessage(id=m.id) for m in state["messages"] if not isinstance(m, SystemMessage)]
        return {"questionIsClear": True, "messages": delete_all, "originalQuery": last_message.content, "rewrittenQuestions": response.questions}

    clarification = response.clarification_needed if response.clarification_needed and len(response.clarification_needed.strip()) > 10 else "我需要更多信息来理解你的问题。"
    return {"questionIsClear": False, "messages": [AIMessage(content=clarification)]}

def request_clarification(state: State):
    return {}

def route_after_rewrite(state: State) -> Literal["request_clarification", "agent"]:
    if not state.get("questionIsClear", False):
        return "request_clarification"
    else:
        return [
                Send("agent", {"question": query, "question_index": idx, "messages": []})
                for idx, query in enumerate(state["rewrittenQuestions"])
            ]

def aggregate_answers(state: State):
    if not state.get("agent_answers"):
        return {"messages": [AIMessage(content="没有生成答案。")]}

    sorted_answers = sorted(state["agent_answers"], key=lambda x: x["index"])

    formatted_answers = ""
    for i, ans in enumerate(sorted_answers, start=1):
        formatted_answers += (f"\n答案 {i}：\n"f"{ans['answer']}\n")

    user_message = HumanMessage(content=f"""原始用户问题：{state["originalQuery"]}\n检索到的答案：{formatted_answers}""")
    synthesis_response = llm.invoke([SystemMessage(content=get_aggregation_prompt()), user_message])
    return {"messages": [AIMessage(content=synthesis_response.content)]}
```

---

#### 智能体子图节点和边
```python
def orchestrator(state: AgentState):
    context_summary = state.get("context_summary", "").strip()
    sys_msg = SystemMessage(content=get_orchestrator_prompt())
    summary_injection = (
        [HumanMessage(content=f"[先前研究的压缩上下文]\n\n{context_summary}")]
        if context_summary else []
    )
    if not state.get("messages"):
        human_msg = HumanMessage(content=state["question"])
        force_search = HumanMessage(content="你必须调用'search_child_chunks'作为回答此问题的第一步。")
        response = llm_with_tools.invoke([sys_msg] + summary_injection + [human_msg, force_search])
        return {"messages": [human_msg, response], "tool_call_count": len(response.tool_calls or []), "iteration_count": 1}

    response = llm_with_tools.invoke([sys_msg] + summary_injection + state["messages"])
    tool_calls = response.tool_calls if hasattr(response, "tool_calls") else []
    return {"messages": [response], "tool_call_count": len(tool_calls) if tool_calls else 0, "iteration_count": 1}

def route_after_orchestrator_call(state: AgentState) -> Literal["tool", "fallback_response", "collect_answer"]:
    iteration = state.get("iteration_count", 0)
    tool_count = state.get("tool_call_count", 0)

    if iteration >= MAX_ITERATIONS or tool_count > MAX_TOOL_CALLS:
        return "fallback_response"

    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []

    if not tool_calls:
        return "collect_answer"
    
    return "tools"

def fallback_response(state: AgentState):
    seen = set()
    unique_contents = []
    for m in state["messages"]:
        if isinstance(m, ToolMessage) and m.content not in seen:
            unique_contents.append(m.content)
            seen.add(m.content)

    context_summary = state.get("context_summary", "").strip()

    context_parts = []
    if context_summary:
        context_parts.append(f"## 压缩的研究上下文（来自先前迭代）\n\n{context_summary}")
    if unique_contents:
        context_parts.append(
            "## 检索到的数据（当前迭代）\n\n" +
            "\n\n".join(f"--- 数据源 {i} ---\n{content}" for i, content in enumerate(unique_contents, 1))
        )

    context_text = "\n\n".join(context_parts) if context_parts else "未从文档中检索到任何数据。"

    prompt_content = (
        f"用户查询：{state.get('question')}\n\n"
        f"{context_text}\n\n"
        f"指令：\n仅使用上述数据提供最佳答案。"
    )
    response = llm.invoke([SystemMessage(content=get_fallback_response_prompt()), HumanMessage(content=prompt_content)])
    return {"messages": [response]}

def should_compress_context(state: AgentState) -> Command[Literal["compress_context", "orchestrator"]]:
    messages = state["messages"]

    new_ids: Set[str] = set()
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                if tc["name"] == "retrieve_parent_chunks":
                    raw = tc["args"].get("parent_id") or tc["args"].get("id") or tc["args"].get("ids") or []
                    if isinstance(raw, str):
                        new_ids.add(f"parent::{raw}")
                    else:
                        new_ids.update(f"parent::{r}" for r in raw)

                elif tc["name"] == "search_child_chunks":
                    query = tc["args"].get("query", "")
                    if query:
                        new_ids.add(f"search::{query}")
            break

    updated_ids = state.get("retrieval_keys", set()) | new_ids

    current_token_messages = estimate_context_tokens(messages)
    current_token_summary = estimate_context_tokens([HumanMessage(content=state.get("context_summary", ""))])
    current_tokens = current_token_messages + current_token_summary

    max_allowed = BASE_TOKEN_THRESHOLD + int(current_token_summary * TOKEN_GROWTH_FACTOR)

    goto = "compress_context" if current_tokens > max_allowed else "orchestrator"
    return Command(update={"retrieval_keys": updated_ids}, goto=goto)

def compress_context(state: AgentState):
    messages = state["messages"]
    existing_summary = state.get("context_summary", "").strip()

    if not messages:
        return {}

    conversation_text = f"用户问题：\n{state.get('question')}\n\n要压缩的对话：\n\n"
    if existing_summary:
        conversation_text += f"[先前的压缩上下文]\n{existing_summary}\n\n"

    for msg in messages[1:]:
        if isinstance(msg, AIMessage):
            tool_calls_info = ""
            if getattr(msg, "tool_calls", None):
                calls = ", ".join(f"{tc['name']}({tc['args']})" for tc in msg.tool_calls)
                tool_calls_info = f" | 工具调用：{calls}"
            conversation_text += f"[助手{tool_calls_info}]\n{msg.content or '(仅工具调用)'}\n\n"
        elif isinstance(msg, ToolMessage):
            tool_name = getattr(msg, "name", "tool")
            conversation_text += f"[工具结果 — {tool_name}]\n{msg.content}\n\n"

    summary_response = llm.invoke([SystemMessage(content=get_context_compression_prompt()), HumanMessage(content=conversation_text)])
    new_summary = summary_response.content

    retrieved_ids: Set[str] = state.get("retrieval_keys", set())
    if retrieved_ids:
        parent_ids = sorted(r for r in retrieved_ids if r.startswith("parent::"))
        search_queries = sorted(r.replace("search::", "") for r in retrieved_ids if r.startswith("search::"))

        block = "\n\n---\n**已执行（不要重复）：**\n"
        if parent_ids:
            block += "已检索的父块：\n" + "\n".join(f"- {p.replace('parent::', '')}" for p in parent_ids) + "\n"
        if search_queries:
            block += "已运行的搜索查询：\n" + "\n".join(f"- {q}" for q in search_queries) + "\n"
        new_summary += block

    return {"context_summary": new_summary, "messages": [RemoveMessage(id=m.id) for m in messages[1:]]}

def collect_answer(state: AgentState):
    last_message = state["messages"][-1]
    is_valid = isinstance(last_message, AIMessage) and last_message.content and not last_message.tool_calls
    answer = last_message.content if is_valid else "无法生成答案。"
    return {
        "final_answer": answer,
        "agent_answers": [{"index": state["question_index"], "question": state["question"], "answer": answer}]
    }
```

**为什么采用这种架构？**
- **摘要**在不压倒 LLM 的情况下维持对话上下文
- **查询重写**确保搜索查询精确且无歧义，智能地使用上下文
- **人工参与**在浪费任何检索资源之前捕获不清晰的查询
- **并行执行**通过`Send` API 为每个子问题同时生成独立的智能体子图
- **上下文压缩**在长检索循环中保持智能体工作记忆的简洁，防止冗余获取
- **后备响应**确保优雅降级 — 即使预算用完，智能体始终返回有用的内容
- **答案收集与聚合**从智能体中提取干净最终答案并聚合成一个连贯的响应
---

### 步骤 10：构建 LangGraph 图

组装包含对话记忆和多智能体架构的完整工作流程图。

```python
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

agent_builder = StateGraph(AgentState)
agent_builder.add_node(orchestrator)
agent_builder.add_node("tools", ToolNode([search_child_chunks, retrieve_parent_chunks]))
agent_builder.add_node(compress_context)
agent_builder.add_node(fallback_response)
agent_builder.add_node(should_compress_context)
agent_builder.add_node(collect_answer)

agent_builder.add_edge(START, "orchestrator")
agent_builder.add_conditional_edges("orchestrator", route_after_orchestrator_call, {"tools": "tools", "fallback_response": "fallback_response", "collect_answer": "collect_answer"})
agent_builder.add_edge("tools", "should_compress_context")
agent_builder.add_edge("compress_context", "orchestrator")
agent_builder.add_edge("fallback_response", "collect_answer")
agent_builder.add_edge("collect_answer", END)
agent_subgraph = agent_builder.compile()

graph_builder = StateGraph(State)
graph_builder.add_node(summarize_history)
graph_builder.add_node(rewrite_query)
graph_builder.add_node(request_clarification)
graph_builder.add_node("agent", agent_subgraph)
graph_builder.add_node(aggregate_answers)

graph_builder.add_edge(START, "summarize_history")
graph_builder.add_edge("summarize_history", "rewrite_query")
graph_builder.add_conditional_edges("rewrite_query", route_after_rewrite)
graph_builder.add_edge("request_clarification", "rewrite_query")
graph_builder.add_edge(["agent"], "aggregate_answers")
graph_builder.add_edge("aggregate_answers", END)

agent_graph = graph_builder.compile(checkpointer=checkpointer, interrupt_before=["request_clarification"])
```

**图架构说明：**

可以在**[此处](./assets/agentic_rag_workflow.png)**查看架构流程图。

**智能体子图**（处理单个问题）：
- START → `orchestrator`（使用工具调用 LLM）
- `orchestrator` → `tools`（如果需要工具调用）或 `fallback_response`（如果预算耗尽）或 `collect_answer`（如果完成）
- `tools` → `should_compress_context`（检查 token 预算）
- `should_compress_context` → `compress_context`（如果超过阈值）或 `orchestrator`（否则）
- `compress_context` → `orchestrator`（使用压缩记忆恢复）
- `fallback_response` → `collect_answer`（打包尽力而为的答案）
- `collect_answer` → END（带有索引的干净最终答案）

**主图**（编排完整工作流）：
- START → `summarize_history`（从历史中提取对话上下文）
- `summarize_history` → `rewrite_query`（用上下文重写查询，检查清晰度）
- `rewrite_query` → `request_clarification`（如果不清晰）或通过`Send`生成并行的`agent`子图（如果清晰）
- `request_clarification` → `rewrite_query`（用户提供澄清后）
- 所有`agent`子图 → `aggregate_answers`（合并所有响应）
- `aggregate_answers` → END（返回最终合成的答案）

---

### 步骤 11：创建聊天界面

构建具有对话持久化和人工参与支持的 Gradio 界面。有关完整的端到端管道 Gradio 界面（包括文档导入），请参阅[project/README.md](./project/README.md)。

> **注意：** 完整的流式支持 — 包括推理步骤和工具调用可见性 — 已在[笔记本](notebooks/agentic_rag.ipynb)和完整的[项目](project/core/chat_interface.py)中实现。下面的示例有意保持最小化 — 仅显示基本的 Gradio 集成模式。

```python
import gradio as gr
import uuid

def create_thread_id():
    """为每个对话生成唯一的线程 ID"""
    return {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 50}

def clear_session():
    """清除线程以开始新对话"""
    global config
    agent_graph.checkpointer.delete_thread(config["configurable"]["thread_id"])
    config = create_thread_id()

def chat(message, history):
    current_state = agent_graph.get_state(config)
    
    if current_state.next:
        agent_graph.update_state(config,{"messages": [HumanMessage(content=message.strip())]})
        result = agent_graph.invoke(None, config)
    else:
        result = agent_graph.invoke({"messages": [HumanMessage(content=message.strip())]}, config)
    
    return result['messages'][-1].content

config = create_thread_id()

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    chatbot.clear(clear_session)
    gr.ChatInterface(fn=chat, chatbot=chatbot)

demo.launch(theme=gr.themes.Citrus())
```

**完成！**你现在拥有一个功能完整的 Agentic RAG 系统，具有对话记忆、层级索引和人工参与的查询澄清功能。

---

## 模块化架构

应用（`project/`文件夹）被组织成模块化组件——每个组件都可以独立替换，而不会破坏系统。

### 📂 项目结构
```
project/
├── app.py                    # 主 Gradio 应用入口点
├── config.py                 # 配置中心（模型、块大小、提供商）
├── core/                     # RAG 系统编排
├── db/                       # 向量数据库和父块存储
├── rag_agent/                # LangGraph 工作流（节点、边、提示词、工具）
└── ui/                       # Gradio 界面
```

关键定制点：LLM 提供商、嵌入模型、分块策略、智能体工作流和系统提示词——均可通过`config.py`或其各自的模块进行配置。

完整文档位于[project/README.md](./project/README.md)。

## 安装与使用

示例 PDF 文件可在此处获取：[javascript](https://www.tutorialspoint.com/javascript/javascript_tutorial.pdf)、[blockchain](https://blockchain-observatory.ec.europa.eu/document/download/1063effa-59cc-4df4-aeee-d2cf94f69178_en?filename=Blockchain_For_Beginners_A_EUBOF_Guide.pdf)、[microservices](https://cdn.studio.f5.com/files/k6fem79d/production/5e4126e1cefa813ab67f9c0b6d73984c27ab1502.pdf)、[fortinet](https://www.commoncriteriaportal.org/files/epfiles/Fortinet%20FortiGate_EAL4_ST_V1.5.pdf(320893)_TMP.pdf)。

### 选项 1：快速开始笔记本（推荐用于测试）

**Google Colab：** 单击本 README 顶部的**在 Colab 中打开**徽章，在文件浏览器中创建一个`docs/`文件夹并上传你的 PDF，使用`pip install -r requirements.txt`安装依赖，然后从头到尾运行所有单元格。

**本地（Jupyter/VSCode）：** 可选择创建并激活虚拟环境，使用`pip install -r requirements.txt`安装依赖，将 PDF 添加到`docs/`，然后从头到尾运行所有单元格。

聊天界面将出现在末尾。

### 选项 2：完整 Python 项目（推荐用于开发）

#### 1. 安装依赖
```bash
# 克隆仓库
git clone https://github.com/GiovanniPasq/agentic-rag-for-dummies
cd agentic-rag-for-dummies

# 可选：创建并激活虚拟环境

# macOS/Linux
python -m venv .venv && source .venv/bin/activate

# Windows
python -m venv .venv && .\.venv\Scripts\activate

# 安装包
pip install -r requirements.txt
```

#### 2. 运行应用
```bash
python project/app.py
```

#### 3. 提问

打开本地 URL（例如`http://127.0.0.1:7860`）开始聊天。

---

### 选项 3：Docker 部署

完整的 Docker 说明和系统要求参见[`project/README.md`](./project/README.md#Docker-Deployment)。

### 对话示例

**带对话记忆：**
```
用户："如何安装 SQL？"
智能体：[从文档中提供安装步骤]

用户："如何更新它？"
智能体：[理解"它"= SQL，提供更新说明]
```

**带查询澄清：**
```
用户："给我讲讲那件事"
智能体："我需要更多信息。你具体在问什么话题？"

用户："PostgreSQL 的安装过程"
智能体：[检索并用具体信息回答]
```

---

## 故障排除

| 领域 | 常见问题 | 建议解决方案 |
|---|---|---|
| **模型选择** | - 响应忽略指令<br>- 工具（检索/搜索）使用不正确<br>- 上下文理解能力差<br>- 幻觉或不完整聚合 | - 使用能力更强的 LLM<br>- 首选 7B+ 模型以获得更好的推理能力<br>- 如果本地模型有限，考虑使用云端模型 |
| **系统提示词行为** | - 模型在未检索文档的情况下回答<br>- 查询重写丢失上下文<br>- 聚合引入幻觉 | - 在系统提示词中明确要求检索<br>- 保持查询重写贴近用户意图 |
| **检索配置** | - 未检索到相关文档<br>- 太多不相关信息 | - 增加检索块数（`k`）或降低相似度阈值以提高召回率<br>- 减少`k`或提高阈值以提高精确度 |
| **块大小/文档拆分** | - 答案缺乏上下文或感觉零碎<br>- 检索速度慢或嵌入成本高 | - 增加块和父块大小以获得更多上下文<br>- 减小块大小以提高速度并降低成本 |
| **上下文压缩** | - 压缩后智能体丢失重要细节<br>- 压缩摘要过于模糊 | - 调整压缩系统提示词<br>- 增加`BASE_TOKEN_THRESHOLD`以延迟压缩<br>- 增加`TOKEN_GROWTH_FACTOR` |
| **智能体配置** | - 智能体过早放弃<br>- 智能体循环过长 | - 对复杂查询增加`MAX_TOOL_CALLS`/`MAX_ITERATIONS`<br>- 对简单查询减少它们以加快速度 |
| **温度与一致性** | - 响应不一致或过于创意<br>- 响应过于刻板或重复 | - 将温度设置为`0`以获得事实性、一致的输出<br>- 对摘要或分析任务略微增加温度 |
| **嵌入模型质量** | - 语义搜索效果差<br>- 对领域特定或多语言文档性能弱 | - 使用更高质量或领域特定的嵌入<br>- 更改嵌入后重新索引所有文档 |

> 💡 **更多故障排除技巧**请参阅[项目 README 故障排除](./project/README.md#troubleshooting)。