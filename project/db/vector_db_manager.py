"""向量数据库管理器模块。

负责 Qdrant 向量数据库的初始化、集合（collection）生命周期管理，
以及 LangChain QdrantVectorStore 的混合检索（Hybrid Retrieval）实例化。

本系统采用「稠密+稀疏」双向量混合搜索架构：
- 稠密向量（Dense）：由 HuggingFace Embeddings（all-mpnet-base-v2）生成，捕捉语义相似度
- 稀疏向量（Sparse）：由 FastEmbedSparse（Qdrant/bm25）生成，捕捉关键词匹配
- 检索模式：RetrievalMode.HYBRID，同时利用两种向量的优势提升召回精度
"""

import config
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels


class VectorDbManager:
    """Qdrant 向量数据库管理器。

    封装了 Qdrant 客户端和嵌入模型的初始化、集合的创建/删除，
    以及 QdrantVectorStore 的获取，为上层 RAG 检索提供统一接口。

    属性：
        __client: Qdrant 客户端实例，使用本地磁盘模式存储（path 模式）
        __dense_embeddings: 稠密向量嵌入模型，用于语义搜索
        __sparse_embeddings: 稀疏向量嵌入模型，用于 BM25 关键词搜索
    """

    __client: QdrantClient
    __dense_embeddings: HuggingFaceEmbeddings
    __sparse_embeddings: FastEmbedSparse

    def __init__(self):
        """初始化向量数据库管理器。

        同时创建三个核心组件：
        1. QdrantClient（本地磁盘模式）— 将向量数据持久化到 config.QDRANT_DB_PATH 指定的目录
        2. HuggingFaceEmbeddings — 加载稠密嵌入模型（默认 all-mpnet-base-v2）
        3. FastEmbedSparse — 加载稀疏嵌入模型（默认 Qdrant/bm25）
        """
        # 使用本地磁盘模式初始化 Qdrant 客户端，数据持久化到本地目录
        self.__client = QdrantClient(path=config.QDRANT_DB_PATH)
        # 初始化稠密向量嵌入模型
        self.__dense_embeddings = HuggingFaceEmbeddings(model_name=config.DENSE_MODEL)
        # 初始化稀疏向量嵌入模型（BM25）
        self.__sparse_embeddings = FastEmbedSparse(model_name=config.SPARSE_MODEL)

    def create_collection(self, collection_name):
        """创建 Qdrant 集合（Collection）。

        如果指定名称的集合尚不存在，则创建一个支持混合检索的新集合，
        包含一个稠密向量字段和一个稀疏向量字段。

        稠密向量参数：
            - 维度：通过 embed_query("test") 动态获取嵌入模型的输出维度
            - 距离度量：余弦距离（COSINE），适合语义相似度计算

        稀疏向量参数：
            - 使用 config.SPARSE_VECTOR_NAME 作为键名（默认 "sparse"）

        Args:
            collection_name: 要创建的集合名称
        """
        # 幂等操作：集合已存在则跳过创建
        if not self.__client.collection_exists(collection_name):
            print(f"Creating collection: {collection_name}...")
            self.__client.create_collection(
                collection_name=collection_name,
                # 稠密向量配置：动态获取维度，使用余弦距离
                vectors_config=qmodels.VectorParams(
                    size=len(self.__dense_embeddings.embed_query("test")),
                    distance=qmodels.Distance.COSINE
                ),
                # 稀疏向量配置：BM25 关键词向量
                sparse_vectors_config={
                    config.SPARSE_VECTOR_NAME: qmodels.SparseVectorParams()
                },
            )
            print(f"✓ Collection created: {collection_name}")
        else:
            print(f"✓ Collection already exists: {collection_name}")

    def delete_collection(self, collection_name):
        """删除指定的 Qdrant 集合。

        如果集合不存在则静默跳过；如果删除过程中发生异常，
        打印警告信息但不中断程序流程。

        Args:
            collection_name: 要删除的集合名称
        """
        try:
            if self.__client.collection_exists(collection_name):
                print(f"Removing existing Qdrant collection: {collection_name}")
                self.__client.delete_collection(collection_name)
        except Exception as e:
            print(f"Warning: could not delete collection {collection_name}: {e}")

    def get_collection(self, collection_name) -> QdrantVectorStore:
        """获取 QdrantVectorStore 实例，用于执行混合检索。

        返回一个配置了稠密+稀疏双向量搜索的 QdrantVectorStore 对象，
        上层可直接调用其 similarity_search 等方法进行文档检索。

        Args:
            collection_name: 要连接的集合名称

        Returns:
            QdrantVectorStore 实例，已配置为混合检索模式

        Raises:
            当集合不存在或连接失败时，打印错误信息并返回 None
        """
        try:
            return QdrantVectorStore(
                client=self.__client,
                collection_name=collection_name,
                embedding=self.__dense_embeddings,
                sparse_embedding=self.__sparse_embeddings,
                retrieval_mode=RetrievalMode.HYBRID,
                sparse_vector_name=config.SPARSE_VECTOR_NAME
            )
        except Exception as e:
            print(f"Unable to get collection {collection_name}: {e}")
