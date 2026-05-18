"""
条件边路由模块

负责根据当前状态决定 LangGraph 图的执行流向。
"""
from typing import Literal
from langgraph.types import Send
from .graph_state import State, AgentState
from config import MAX_ITERATIONS, MAX_TOOL_CALLS


def route_after_rewrite(state: State) -> Literal["request_clarification", "agent"]:
    """查询重写后的路由函数

    根据查询是否清晰，决定是请求用户澄清，还是将重写后的查询
    分发给并行执行的智能体子图。

    Args:
        state: 主图状态，包含 questionIsClear 和 rewrittenQuestions 字段

    Returns:
        如果查询不清晰，返回 "request_clarification" 节点；
        否则返回并行发送的 agent 子图列表，每个子图处理一个重写后的查询。
    """
    if not state.get("questionIsClear", False):
        return "request_clarification"
    else:
        return [
                Send("agent", {"question": query, "question_index": idx, "messages": []})
                for idx, query in enumerate(state["rewrittenQuestions"])
            ]


def route_after_orchestrator_call(state: AgentState) -> Literal["tool", "fallback_response", "collect_answer"]:
    """编排器调用后的路由函数

    根据迭代次数和工具调用次数判断是否达到预算上限，
    决定是继续调用工具、生成降级响应，还是收集答案。

    Args:
        state: 智能体子图状态，包含 iteration_count 和 tool_call_count 字段

    Returns:
        如果达到预算上限，返回 "fallback_response"；
        如果最后一条消息没有工具调用，返回 "collect_answer"；
        否则返回 "tools" 继续执行工具调用。
    """
    iteration = state.get("iteration_count", 0)
    tool_count = state.get("tool_call_count", 0)

    if iteration >= MAX_ITERATIONS or tool_count > MAX_TOOL_CALLS:
        return "fallback_response"

    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []

    if not tool_calls:
        return "collect_answer"
    
    return "tools"
