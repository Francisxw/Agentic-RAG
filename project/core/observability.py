"""
可观测性模块 — Langfuse 集成

负责初始化 Langfuse 客户端和回调处理器，为 Agentic RAG 系统的
LLM 调用、工具使用和图执行提供追踪与可观测能力。
"""

import logging
import config

# 模块级日志记录器，用于记录可观测性初始化的警告和错误信息
logger = logging.getLogger(__name__)


class Observability:
    """Langfuse 可观测性管理器

    封装 Langfuse 客户端的初始化、认证检查和生命周期管理。
    当配置中未启用 Langfuse 或 API 密钥缺失时， gracefully 降级，
    不影响 RAG 系统的正常运行。
    """

    def __init__(self):
        """初始化 Langfuse 可观测性组件

        执行流程：
        1. 检查配置开关（config.LANGFUSE_ENABLED）
        2. 验证 API 公钥和私钥是否已配置
        3. 尝试创建 Langfuse 客户端并验证认证
        4. 创建 LangChain 回调处理器，供 LangGraph 图使用
        5. 任何步骤失败都会 gracefully 禁用可观测性并记录警告
        """
        # 是否启用了 Langfuse 追踪（由环境变量 LANGFUSE_ENABLED 控制）
        self._enabled = config.LANGFUSE_ENABLED
        # Langfuse 回调处理器 — 注入到 LangGraph 节点中以捕获追踪数据
        self._handler = None
        # Langfuse 客户端实例 — 用于认证检查和数据刷新
        self._client = None

        # 如果配置中未启用，直接返回，不初始化任何组件
        if not self._enabled:
            return

        # 检查 API 密钥是否已配置，缺失则禁用并记录警告
        if not config.LANGFUSE_PUBLIC_KEY or not config.LANGFUSE_SECRET_KEY:
            logger.warning("Langfuse enabled but API keys are missing — skipping")
            self._enabled = False
            return

        try:
            # 延迟导入：仅在需要时才加载 Langfuse，避免未安装时崩溃
            from langfuse import get_client
            from langfuse.langchain import CallbackHandler

            # 创建 Langfuse 客户端实例
            self._client = get_client()

            # 验证客户端认证是否成功
            if self._client.auth_check():
                print("Langfuse client is authenticated and ready!")
            else:
                print("Authentication failed. Please check your credentials and host.")
                self._enabled = False
                return

            # 创建 LangChain 回调处理器，后续注入到 LangGraph 图中
            self._handler = CallbackHandler()
        except Exception as exc:
            # 初始化失败时记录警告并禁用可观测性，不影响主流程
            logger.warning("Could not initialize Langfuse: %s", exc)
            self._enabled = False

    def get_handler(self):
        """获取 Langfuse 回调处理器

        返回用于 LangChain/LangGraph 的回调处理器实例。
        如果可观测性未启用，返回 None。

        Returns:
            CallbackHandler | None: Langfuse 回调处理器，未启用时返回 None
        """
        return self._handler

    def flush(self):
        """强制刷新未发送的追踪数据到 Langfuse 服务端

        在应用关闭或对话结束时调用，确保所有缓冲的追踪事件
        都被发送到 Langfuse，避免数据丢失。
        """
        if self._client is not None:
            try:
                self._client.flush()
            except Exception:
                # 刷新失败不影响主流程，静默忽略
                pass