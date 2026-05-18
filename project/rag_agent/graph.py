from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode
from functools import partial

from .graph_state import State
from .nodes import *
from .edges import *


def create_agent_graph(llm, tools_list):
    """构建 Agentic RAG 的完整 LangGraph 工作流图。

    该函数创建两层嵌套图结构：
    - 智能体子图（agent_subgraph）：处理单个查询的检索和推理循环
    - 主图（agent_graph）：编排对话理解、查询重写、并行智能体执行和答案聚合

    Args:
        llm: LangChain LLM 实例（已绑定工具）
        tools_list: 智能体可用的工具列表（如 search_child_chunks, retrieve_parent_chunks）

    Returns:
        编译后的 LangGraph StateGraph，带有检查点和澄清中断
    """
    # 将工具绑定到 LLM，使其具备工具调用能力
    llm_with_tools = llm.bind_tools(tools_list)
    tool_node = ToolNode(tools_list)

    # 内存检查点：支持对话持久化和人工澄清中断
    checkpointer = InMemorySaver()

    print("Compiling agent graph...")

    # ========== 智能体子图：处理单个查询的检索循环 ==========
    # 每个子图负责一个独立查询的完整研究流程：编排 → 工具调用 → 压缩 → 收集
    agent_builder = StateGraph(AgentState)
    agent_builder.add_node("orchestrator", partial(orchestrator, llm_with_tools=llm_with_tools))       # 核心编排节点：决定是否需要检索
    agent_builder.add_node("tools", tool_node)                                                          # 执行工具调用（搜索子块、检索父块）
    agent_builder.add_node("compress_context", partial(compress_context, llm=llm))                      # 压缩对话上下文以节省 token
    agent_builder.add_node("fallback_response", partial(fallback_response, llm=llm))                    # 预算耗尽时的降级响应
    agent_builder.add_node(should_compress_context)                                                     # 条件节点：判断是否需要压缩上下文
    agent_builder.add_node(collect_answer)                                                              # 收集智能体的最终答案

    # 子图边连接
    agent_builder.add_edge(START, "orchestrator")
    agent_builder.add_conditional_edges("orchestrator", route_after_orchestrator_call,
        {"tools": "tools", "fallback_response": "fallback_response", "collect_answer": "collect_answer"})
    agent_builder.add_edge("tools", "should_compress_context")
    agent_builder.add_edge("compress_context", "orchestrator")
    agent_builder.add_edge("fallback_response", "collect_answer")
    agent_builder.add_edge("collect_answer", END)

    # 编译子图，供主图作为嵌套节点调用
    agent_subgraph = agent_builder.compile()

    # ========== 主图：编排完整工作流 ==========
    # 主图负责：对话摘要 → 查询重写 → 并行智能体执行 → 答案聚合
    graph_builder = StateGraph(State)
    graph_builder.add_node("summarize_history", partial(summarize_history, llm=llm))       # 从对话历史中提取上下文摘要
    graph_builder.add_node("rewrite_query", partial(rewrite_query, llm=llm))              # 重写查询以优化检索
    graph_builder.add_node(request_clarification)                                          # 暂停并向用户请求澄清
    graph_builder.add_node("agent", agent_subgraph)                                        # 嵌套智能体子图（可并行执行多个）
    graph_builder.add_node("aggregate_answers", partial(aggregate_answers, llm=llm))       # 将多个智能体答案聚合成最终响应

    # 主图边连接
    graph_builder.add_edge(START, "summarize_history")
    graph_builder.add_edge("summarize_history", "rewrite_query")
    graph_builder.add_conditional_edges("rewrite_query", route_after_rewrite)              # 根据查询清晰度路由
    graph_builder.add_edge("request_clarification", "rewrite_query")
    graph_builder.add_edge(["agent"], "aggregate_answers")
    graph_builder.add_edge("aggregate_answers", END)

    # 编译主图：启用检查点支持，并在 request_clarification 节点前中断以等待人工输入
    agent_graph = graph_builder.compile(checkpointer=checkpointer, interrupt_before=["request_clarification"])

    print("✓ Agent graph compiled successfully.")
    return agent_graph

