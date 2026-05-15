import uuid
from langchain_ollama import ChatOllama
import config
from db.vector_db_manager import VectorDbManager
from db.parent_store_manager import ParentStoreManager
from document_chunker import DocumentChuncker
from rag_agent.tools import ToolFactory
from rag_agent.graph import create_agent_graph
from core.observability import Observability


class RAGSystem:
    """Agentic RAG 系统入口

    作为整个 RAG 系统的统一入口，编排文档索引、向量检索、
    LangGraph 智能体工作流和可观测性组件。
    外部调用方通过此类与整个系统交互。
    """

    def __init__(self, collection_name=config.CHILD_COLLECTION):
        """初始化 RAG 系统各组件

        Args:
            collection_name: 向量数据库中存储子块的集合名称，默认使用配置文件中的 CHILD_COLLECTION
        """
        self.collection_name = collection_name
        self.vector_db = VectorDbManager()
        self.parent_store = ParentStoreManager()
        self.chunker = DocumentChuncker()
        self.observability = Observability()
        self.agent_graph = None
        self.thread_id = str(uuid.uuid4())
        self.recursion_limit = config.GRAPH_RECURSION_LIMIT

    def initialize(self):
        """初始化 RAG 系统

        创建向量数据库集合、构建 LLM 实例、注册智能体工具、
        并编译 LangGraph 智能体图。此方法应在首次使用系统前调用。
        """
        self.vector_db.create_collection(self.collection_name)
        collection = self.vector_db.get_collection(self.collection_name)

        llm = ChatOllama(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)
        tools = ToolFactory(collection).create_tools()
        self.agent_graph = create_agent_graph(llm, tools)

    def get_config(self):
        """获取 LangGraph 图的运行配置

        返回包含 thread_id、recursion_limit 和可选的可观测性回调的配置字典，
        可直接传递给 agent_graph.invoke() 或 agent_graph.stream()。

        Returns:
            配置字典，包含 configurable（线程 ID）和 recursion_limit（递归深度限制），
            若启用了可观测性则额外包含 callbacks
        """
        cfg = {"configurable": {"thread_id": self.thread_id}, "recursion_limit": self.recursion_limit}
        handler = self.observability.get_handler()
        if handler:
            cfg["callbacks"] = [handler]
        return cfg

    def reset_thread(self):
        """重置对话线程

        删除当前线程的对话历史（从 checkpointer 中），
        并生成新的 thread_id 以开始全新的对话。
        """
        try:
            self.agent_graph.checkpointer.delete_thread(self.thread_id)
        except Exception as e:
            print(f"Warning: Could not delete thread {self.thread_id}: {e}")
        self.thread_id = str(uuid.uuid4())