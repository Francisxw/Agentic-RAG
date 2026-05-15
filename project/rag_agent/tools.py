from typing import List
from langchain_core.tools import tool
from db.parent_store_manager import ParentStoreManager


class ToolFactory:
    """工具工厂类

    负责创建和管理 Agentic RAG 系统的检索工具。
    采用父/子块（Parent/Child Chunk）层级索引策略：
    - 子块（小片段）用于精确相似度搜索
    - 父块（大段落）用于提供丰富上下文以生成答案

    Args:
        collection: Qdrant 向量数据库集合，用于存储和搜索子块嵌入
    """

    def __init__(self, collection):
        """初始化工具工厂

        Args:
            collection: Qdrant 向量数据库集合（存储子块），
                        支持混合搜索（dense + sparse）和相似度查询
        """
        self.collection = collection
        self.parent_store_manager = ParentStoreManager()
    
    def _search_child_chunks(self, query: str, limit: int) -> str:
        """在向量数据库中搜索最相关的 K 个子块（小片段）。

        子块是从父块拆分出的固定大小片段，适合精确语义匹配。
        搜索使用相似度阈值（0.7）过滤低相关结果。

        Args:
            query: 搜索查询字符串
            limit: 返回的最大结果数量

        Returns:
            格式化后的相关子块文本，以 "\\n\\n" 分隔。
            每条结果包含：父块 ID、源文件名、子块内容。
            若无相关结果返回 "NO_RELEVANT_CHUNKS"，
            若发生异常返回 "RETRIEVAL_ERROR: <错误信息>"。
        """
        try:
            # 执行混合相似度搜索，返回按相关性排序的文档列表
            results = self.collection.similarity_search(query, k=limit, score_threshold=0.7)
            if not results:
                return "NO_RELEVANT_CHUNKS"

            # 将搜索结果格式化为可读文本
            return "\n\n".join([
                f"Parent ID: {doc.metadata.get('parent_id', '')}\n"
                f"File Name: {doc.metadata.get('source', '')}\n"
                f"Content: {doc.page_content.strip()}"
                for doc in results
            ])            

        except Exception as e:
            return f"RETRIEVAL_ERROR: {str(e)}"
    
    def _retrieve_many_parent_chunks(self, parent_ids: List[str]) -> str:
        """根据 ID 列表批量检索完整的父块（大段落）。

        父块是基于 Markdown 标题结构划分的大段内容，
        包含更丰富的上下文信息，用于生成高质量答案。

        Args:
            parent_ids: 父块 ID 列表（也支持传入单个字符串）

        Returns:
            格式化后的父块文本，以 "\\n\\n" 分隔。
            每条结果包含：父块 ID、源文件名、父块内容。
            若无相关结果返回 "NO_PARENT_DOCUMENTS"，
            若发生异常返回 "PARENT_RETRIEVAL_ERROR: <错误信息>"。
        """
        try:
            # 统一格式：确保 parent_ids 为列表
            ids = [parent_ids] if isinstance(parent_ids, str) else list(parent_ids)
            raw_parents = self.parent_store_manager.load_content_many(ids)
            if not raw_parents:
                return "NO_PARENT_DOCUMENTS"

            # 将批量检索结果格式化为可读文本
            return "\n\n".join([
                f"Parent ID: {doc.get('parent_id', 'n/a')}\n"
                f"File Name: {doc.get('metadata', {}).get('source', 'unknown')}\n"
                f"Content: {doc.get('content', '').strip()}"
                for doc in raw_parents
            ])            

        except Exception as e:
            return f"PARENT_RETRIEVAL_ERROR: {str(e)}"
    
    def _retrieve_parent_chunks(self, parent_id: str) -> str:
        """根据 ID 检索单个完整的父块（大段落）。

        与 _retrieve_many_parent_chunks 不同，此方法仅检索单个父块，
        返回格式也相应简化为单条结果。

        Args:
            parent_id: 要检索的父块唯一标识符

        Returns:
            格式化后的父块文本，包含：父块 ID、源文件名、父块内容。
            若未找到返回 "NO_PARENT_DOCUMENT"，
            若发生异常返回 "PARENT_RETRIEVAL_ERROR: <错误信息>"。
        """
        try:
            parent = self.parent_store_manager.load_content(parent_id)
            if not parent:
                return "NO_PARENT_DOCUMENT"

            # 将检索结果格式化为可读文本
            return (
                f"Parent ID: {parent.get('parent_id', 'n/a')}\n"
                f"File Name: {parent.get('metadata', {}).get('source', 'unknown')}\n"
                f"Content: {parent.get('content', '').strip()}"
            )          

        except Exception as e:
            return f"PARENT_RETRIEVAL_ERROR: {str(e)}"
    
    def create_tools(self) -> List:
        """创建并返回智能体可用的工具列表。

        将内部方法包装为 LangChain Tool 对象，
        使其可以被 LangGraph 智能体通过工具调用来执行检索操作。

        Returns:
            包含两个 LangChain Tool 对象的列表：
            1. search_child_chunks — 搜索相关子块
            2. retrieve_parent_chunks — 检索完整父块
        """
        search_tool = tool("search_child_chunks")(self._search_child_chunks)
        retrieve_tool = tool("retrieve_parent_chunks")(self._retrieve_parent_chunks)
        
        return [search_tool, retrieve_tool]