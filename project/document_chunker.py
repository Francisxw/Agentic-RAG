"""
文档分块模块

负责将 Markdown 格式的文档拆分为层级索引结构（父块 + 子块），
用于 Agentic RAG 系统中的混合向量检索。

分块策略：
  1. 基于 Markdown 标题层级（H1/H2/H3）进行首次分割 → 父块
  2. 合并过小的父块，拆分过大的父块 → 规范化父块
  3. 将规范化后的父块进一步切分为固定大小的子块 → 子块（存入向量数据库）

父块用于提供丰富上下文，子块用于精确检索，二者配合实现"小块搜索精确、大块上下文丰富"的效果。
"""

import os
import glob
import config
from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


class DocumentChuncker:
    """文档分块器

    使用两级分块策略处理 Markdown 文档：
    - 父块（Parent Chunks）：按 Markdown 标题层级分割，保留上下文结构
    - 子块（Child Chunks）：从父块中进一步切分为固定大小片段，用于向量检索

    属性：
        __parent_splitter: 基于标题层级的 Markdown 文档分割器
        __child_splitter: 基于字符长度的递归文本分割器
        __min_parent_size: 父块最小字符数，小于此值将合并到相邻块
        __max_parent_size: 父块最大字符数，大于此值将拆分
    """

    def __init__(self):
        """初始化分块器，配置父块和子块分割器"""
        # 父块分割器：按 Markdown 标题层级（H1/H2/H3）分割文档
        self.__parent_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=config.HEADERS_TO_SPLIT_ON,  # 分割头：[("#", "H1"), ("##", "H2"), ("###", "H3")]
            strip_headers=False  # 保留标题信息在元数据中
        )
        # 子块分割器：将父块进一步切分为固定大小的片段
        self.__child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHILD_CHUNK_SIZE,       # 子块大小：500 字符
            chunk_overlap=config.CHILD_CHUNK_OVERLAP  # 子块重叠：100 字符
        )
        # 父块大小约束
        self.__min_parent_size = config.MIN_PARENT_SIZE   # 最小父块大小：2000 字符
        self.__max_parent_size = config.MAX_PARENT_SIZE   # 最大父块大小：4000 字符

    def create_chunks(self, path_dir=config.MARKDOWN_DIR):
        """批量处理指定目录下的所有 Markdown 文档

        遍历目录中所有 .md 文件，逐个调用 create_chunks_single 进行分块，
        最终汇总所有父块和子块。

        参数：
            path_dir: Markdown 文件所在目录路径，默认为配置中的 MARKDOWN_DIR

        返回：
            tuple: (all_parent_chunks, all_child_chunks)
                - all_parent_chunks: 所有文档的父块列表
                - all_child_chunks: 所有文档的子块列表
        """
        all_parent_chunks, all_child_chunks = [], []

        # 按文件名排序，确保处理顺序一致
        for doc_path_str in sorted(glob.glob(os.path.join(path_dir, "*.md"))):
            doc_path = Path(doc_path_str)
            parent_chunks, child_chunks = self.create_chunks_single(doc_path)
            all_parent_chunks.extend(parent_chunks)
            all_child_chunks.extend(child_chunks)

        return all_parent_chunks, all_child_chunks

    def create_chunks_single(self, md_path):
        """处理单个 Markdown 文档的分块流程

        完整分块流水线：
        1. 按标题层级分割 → 初步父块
        2. 合并过小的父块 → 避免碎片化
        3. 拆分过大的父块 → 控制最大尺寸
        4. 清理剩余的小块 → 确保所有父块达到最小尺寸要求
        5. 为每个父块生成子块 → 存入向量检索

        参数：
            md_path: Markdown 文件路径（Path 对象或字符串）

        返回：
            tuple: (all_parent_chunks, all_child_chunks)
        """
        doc_path = Path(md_path)

        # 读取文档并按标题层级分割为初步父块
        with open(doc_path, "r", encoding="utf-8") as f:
            parent_chunks = self.__parent_splitter.split_text(f.read())

        # 流水线处理：合并小块 → 拆大块 → 清理残余小块
        merged_parents = self.__merge_small_parents(parent_chunks)
        split_parents = self.__split_large_parents(merged_parents)
        cleaned_parents = self.__clean_small_chunks(split_parents)

        # 为每个规范化后的父块生成子块，并补充元数据
        all_parent_chunks, all_child_chunks = [], []
        self.__create_child_chunks(all_parent_chunks, all_child_chunks, cleaned_parents, doc_path)
        return all_parent_chunks, all_child_chunks

    def __merge_small_parents(self, chunks):
        """合并过小的父块

        将字符数未达到最小阈值（MIN_PARENT_SIZE）的连续块合并在一起，
        避免产生过于碎片化的上下文。合并时同时合并元数据，
        同一键的值用 " -> " 连接以保留完整的标题路径信息。

        参数：
            chunks: 初步分割后的父块列表

        返回：
            list: 合并后的父块列表
        """
        if not chunks:
            return []

        merged, current = [], None

        for chunk in chunks:
            if current is None:
                current = chunk
            else:
                # 将当前块内容追加到累积块
                current.page_content += "\n\n" + chunk.page_content
                # 合并元数据：同一键的值用 " -> " 连接，保留标题层级路径
                for k, v in chunk.metadata.items():
                    if k in current.metadata:
                        current.metadata[k] = f"{current.metadata[k]} -> {v}"
                    else:
                        current.metadata[k] = v

            # 累积块达到最小尺寸后，将其加入结果并重置
            if len(current.page_content) >= self.__min_parent_size:
                merged.append(current)
                current = None

        # 处理最后一个未达阈值的累积块
        if current:
            if merged:
                # 追加到最后一个已合并的块
                merged[-1].page_content += "\n\n" + current.page_content
                for k, v in current.metadata.items():
                    if k in merged[-1].metadata:
                        merged[-1].metadata[k] = f"{merged[-1].metadata[k]} -> {v}"
                    else:
                        merged[-1].metadata[k] = v
            else:
                # 如果没有任何已合并的块，直接追加（即使未达最小尺寸）
                merged.append(current)

        return merged

    def __split_large_parents(self, chunks):
        """拆分过大的父块

        对于字符数超过最大阈值（MAX_PARENT_SIZE）的父块，
        使用 RecursiveCharacterTextSplitter 将其拆分为多个不超过最大尺寸的子块。

        参数：
            chunks: 待检查的父块列表

        返回：
            list: 拆分后的父块列表（所有块均不超过 MAX_PARENT_SIZE）
        """
        split_chunks = []

        for chunk in chunks:
            if len(chunk.page_content) <= self.__max_parent_size:
                # 未超过上限，直接保留
                split_chunks.append(chunk)
            else:
                # 超过上限，使用子块分割器进行拆分
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.__max_parent_size,
                    chunk_overlap=config.CHILD_CHUNK_OVERLAP  # 保持与子块一致的重叠策略
                )
                sub_chunks = splitter.split_documents([chunk])
                split_chunks.extend(sub_chunks)

        return split_chunks

    def __clean_small_chunks(self, chunks):
        """清理剩余的小块

        在合并和拆分之后，仍可能存在未达最小尺寸的块。
        此方法将小块的内容追加到前一个块或下一个块中，
        确保最终结果中所有块都尽可能达到最小尺寸要求。

        处理策略：
        - 如果前面已有合并的块 → 追加到前一个块
        - 否则如果后面还有块 → 追加到下一个块
        - 否则（最后一个且前面无块）→ 保留原样（避免丢失内容）

        参数：
            chunks: 待清理的父块列表

        返回：
            list: 清理后的父块列表
        """
        cleaned = []

        for i, chunk in enumerate(chunks):
            if len(chunk.page_content) < self.__min_parent_size:
                # 当前块过小，需要合并
                if cleaned:
                    # 追加到已清理的最后一个块
                    cleaned[-1].page_content += "\n\n" + chunk.page_content
                    for k, v in chunk.metadata.items():
                        if k in cleaned[-1].metadata:
                            cleaned[-1].metadata[k] = f"{cleaned[-1].metadata[k]} -> {v}"
                        else:
                            cleaned[-1].metadata[k] = v
                elif i < len(chunks) - 1:
                    # 追加到下一个块（原地修改列表中的元素）
                    chunks[i + 1].page_content = chunk.page_content + "\n\n" + chunks[i + 1].page_content
                    for k, v in chunk.metadata.items():
                        if k in chunks[i + 1].metadata:
                            chunks[i + 1].metadata[k] = f"{v} -> {chunks[i + 1].metadata[k]}"
                        else:
                            chunks[i + 1].metadata[k] = v
                else:
                    # 最后一个块且前面无块可合并 → 保留（宁可保留小块也不丢失内容）
                    cleaned.append(chunk)
            else:
                # 尺寸达标，直接加入结果
                cleaned.append(chunk)

        return cleaned

    def __create_child_chunks(self, all_parent_pairs, all_child_chunks, parent_chunks, doc_path):
        """为每个父块生成子块并补充元数据

        对每个规范化后的父块：
        1. 生成唯一的父块 ID（格式：{文档名}_parent_{索引}）
        2. 补充 source（源 PDF 文件名）和 parent_id 元数据
        3. 将父块加入父块列表，子块加入子块列表

        参数：
            all_parent_pairs: 父块累积列表（传入空列表，方法内部追加）
            all_child_chunks: 子块累积列表（传入空列表，方法内部追加）
            parent_chunks: 规范化后的父块列表
            doc_path: 源文档路径（用于生成元数据）
        """
        for i, p_chunk in enumerate(parent_chunks):
            # 生成父块唯一标识符
            parent_id = f"{doc_path.stem}_parent_{i}"
            # 补充元数据：source 指向原始 PDF 文件，parent_id 用于后续反向检索父块
            p_chunk.metadata.update({"source": str(doc_path.stem) + ".pdf", "parent_id": parent_id})

            # 将父块加入累积列表（格式：(parent_id, Document) 元组）
            all_parent_pairs.append((parent_id, p_chunk))
            # 将父块切分为子块并加入子块累积列表
            all_child_chunks.extend(self.__child_splitter.split_documents([p_chunk]))
