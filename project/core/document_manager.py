"""
文档管理模块

负责文档的完整生命周期管理：
- PDF 转 Markdown 转换
- 层级分块（父块/子块）
- 向量数据库索引
- 父块文件存储
"""

from pathlib import Path
import shutil
import config
# pdfs_to_markdowns: PDF → Markdown 批量转换
# clear_directory_contents: 清空目录内容
from utils import pdfs_to_markdowns, clear_directory_contents


class DocumentManager:
    """文档管理器

    管理文档从上传到可检索的完整流程：
    1. 接收 PDF 或 Markdown 文件
    2. 将 PDF 转换为 Markdown（如需要）
    3. 通过 RAG 系统的 chunker 进行父子分块
    4. 将子块索引到向量数据库，父块保存到文件系统
    5. 跳过已存在的文档以避免重复索引
    """

    def __init__(self, rag_system):
        """初始化文档管理器

        Args:
            rag_system: RAG 系统实例，提供 chunker、vector_db、parent_store 等组件
        """
        self.rag_system = rag_system
        # Markdown 文件存储目录
        self.markdown_dir = Path(config.MARKDOWN_DIR)
        self.markdown_dir.mkdir(parents=True, exist_ok=True)

    def add_documents(self, document_paths, progress_callback=None):
        """添加文档到 RAG 系统

        对每个文档执行：PDF→Markdown 转换 → 父子分块 → 向量索引 → 父块存储。
        已存在的 Markdown 文件会被跳过。处理失败的文档也会被跳过，
        不会中断整个批次。

        Args:
            document_paths: 文件路径，可以是单个字符串或路径列表
            progress_callback: 可选的进度回调函数，签名 callback(progress: float, message: str)

        Returns:
            tuple[int, int]: (成功添加的文档数, 跳过的文档数)
        """
        # 空输入直接返回
        if not document_paths:
            return 0, 0

        # 统一为列表格式
        document_paths = [document_paths] if isinstance(document_paths, str) else document_paths
        # 仅保留 .pdf 和 .md 文件
        document_paths = [p for p in document_paths if p and Path(p).suffix.lower() in [".pdf", ".md"]]

        if not document_paths:
            return 0, 0

        added = 0
        skipped = 0

        for i, doc_path in enumerate(document_paths):
            # 通知调用方当前进度
            if progress_callback:
                progress_callback((i + 1) / len(document_paths), f"Processing {Path(doc_path).name}")

            doc_name = Path(doc_path).stem
            md_path = self.markdown_dir / f"{doc_name}.md"

            # 如果该文档的 Markdown 已存在，跳过以避免重复索引
            if md_path.exists():
                skipped += 1
                continue

            try:
                # 根据文件类型选择处理方式
                if Path(doc_path).suffix.lower() == ".md":
                    # 直接复制 Markdown 文件
                    shutil.copy(doc_path, md_path)
                else:
                    # PDF 文件：调用转换工具生成 Markdown
                    pdfs_to_markdowns(str(doc_path), overwrite=False)

                # 对 Markdown 文档进行父子分块
                parent_chunks, child_chunks = self.rag_system.chunker.create_chunks_single(md_path)

                # 如果没有生成任何子块，跳过该文档
                if not child_chunks:
                    skipped += 1
                    continue

                # 将子块索引到向量数据库，父块保存到文件系统
                collection = self.rag_system.vector_db.get_collection(self.rag_system.collection_name)
                collection.add_documents(child_chunks)
                self.rag_system.parent_store.save_many(parent_chunks)

                added += 1

            except Exception as e:
                # 处理失败时记录错误并跳过该文档，不影响其他文档
                print(f"Error processing {doc_path}: {e}")
                skipped += 1

        return added, skipped

    def get_markdown_files(self):
        """获取已索引的 Markdown 文件列表

        Returns:
            list[str]: 已索引的文档名称列表（统一以 .pdf 后缀显示，
                       因为原始上传的都是 PDF 文件）
        """
        if not self.markdown_dir.exists():
            return []
        # 返回所有 .md 文件名，将后缀替换为 .pdf 以匹配原始上传文件名
        return sorted([p.name.replace(".md", ".pdf") for p in self.markdown_dir.glob("*.md")])

    def clear_all(self):
        """清除所有文档数据

        清空以下所有存储：
        - Markdown 文件目录
        - 父块文件存储
        - 向量数据库集合（重建空集合）
        """
        # 确保 Markdown 目录存在后清空其内容
        self.markdown_dir.mkdir(parents=True, exist_ok=True)
        clear_directory_contents(self.markdown_dir)

        # 清空父块存储
        self.rag_system.parent_store.clear_store()
        # 删除旧向量集合并重建空集合
        self.rag_system.vector_db.delete_collection(self.rag_system.collection_name)
        self.rag_system.vector_db.create_collection(self.rag_system.collection_name)
