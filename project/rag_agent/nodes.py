from typing import Literal, Set
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage, AIMessage, ToolMessage
from langgraph.types import Command
from .graph_state import State, AgentState
from .schemas import QueryAnalysis
from .prompts import *
from utils import estimate_context_tokens
from config import BASE_TOKEN_THRESHOLD, TOKEN_GROWTH_FACTOR


def summarize_history(state: State, llm):
    """从对话历史中提取上下文摘要，用于维持多轮对话的连贯性。
    
    如果消息少于 4 条则跳过摘要（对话太短无需摘要）。
    同时重置 agent_answers，为新的查询周期做准备。
    """
    # 对话太短，无需摘要
    if len(state["messages"]) < 4:
        return {"conversation_summary": ""}
    
    # 筛选出人类消息和助手回复（排除工具调用相关的消息）
    relevant_msgs = [
        msg for msg in state["messages"][:-1]
        if isinstance(msg, (HumanMessage, AIMessage)) and not getattr(msg, "tool_calls", None)
    ]

    if not relevant_msgs:
        return {"conversation_summary": ""}
    
    # 构建最近 6 条相关消息的对话文本
    conversation = "对话历史：\n"
    for msg in relevant_msgs[-6:]:
        role = "用户" if isinstance(msg, HumanMessage) else "助手"
        conversation += f"{role}: {msg.content}\n"

    # 调用 LLM 生成对话摘要
    summary_response = llm.with_config(temperature=0.2).invoke([SystemMessage(content=get_conversation_summary_prompt()), HumanMessage(content=conversation)])
    # 返回摘要，并重置 agent_answers（新查询周期）
    return {"conversation_summary": summary_response.content, "agent_answers": [{"__reset__": True}]}

def rewrite_query(state: State, llm):
    """重写用户查询以优化检索。
    
    融合对话上下文，将查询重写为自包含的形式。
    如果查询不清晰，返回澄清请求。
    """
    last_message = state["messages"][-1]
    conversation_summary = state.get("conversation_summary", "")

    # 构建包含对话上下文和用户查询的输入
    context_section = (f"对话上下文：\n{conversation_summary}\n" if conversation_summary.strip() else "") + f"用户查询：\n{last_message.content}\n"

    # 使用结构化输出解析查询分析结果
    llm_with_structure = llm.with_config(temperature=0.1).with_structured_output(QueryAnalysis)
    response = llm_with_structure.invoke([SystemMessage(content=get_rewrite_query_prompt()), HumanMessage(content=context_section)])

    # 查询清晰：删除旧消息，返回重写后的查询列表
    if response.questions and response.is_clear:
        delete_all = [RemoveMessage(id=m.id) for m in state["messages"] if not isinstance(m, SystemMessage)]
        return {"questionIsClear": True, "messages": delete_all, "originalQuery": last_message.content, "rewrittenQuestions": response.questions}

    # 查询不清晰：返回澄清请求
    clarification = response.clarification_needed if response.clarification_needed and len(response.clarification_needed.strip()) > 10 else "我需要更多信息来理解你的问题。"
    return {"questionIsClear": False, "messages": [AIMessage(content=clarification)]}

def request_clarification(state: State):
    """澄清节点：暂停并等待用户输入。
    
    此节点本身不执行任何逻辑，仅作为图的中间节点，
    配合 checkpointer 在编译时设置 interrupt_before 实现人工中断。
    """
    return {}


# ========== 智能体子图节点 ==========

def orchestrator(state: AgentState, llm_with_tools):
    """核心编排节点：决定是否需要检索工具。
    
    作为智能体的"大脑"，它根据压缩上下文判断当前信息是否充足，
    不足则调用工具搜索，充足则直接生成答案。
    """
    context_summary = state.get("context_summary", "").strip()
    sys_msg = SystemMessage(content=get_orchestrator_prompt())
    # 将压缩上下文注入到系统提示中
    summary_injection = (
        [HumanMessage(content=f"[先前研究的压缩上下文]\n\n{context_summary}")]
        if context_summary else []
    )
    # 首次调用：注入人类消息和强制搜索指令
    if not state.get("messages"):
        human_msg = HumanMessage(content=state["question"])
        force_search = HumanMessage(content="你必须调用'search_child_chunks'作为回答此问题的第一步。")
        response = llm_with_tools.invoke([sys_msg] + summary_injection + [human_msg, force_search])
        return {"messages": [human_msg, response], "tool_call_count": len(response.tool_calls or []), "iteration_count": 1}

    # 后续迭代：基于已有消息继续推理
    response = llm_with_tools.invoke([sys_msg] + summary_injection + state["messages"])
    tool_calls = response.tool_calls if hasattr(response, "tool_calls") else []
    return {"messages": [response], "tool_call_count": len(tool_calls) if tool_calls else 0, "iteration_count": 1}

def fallback_response(state: AgentState, llm):
    """预算耗尽时的降级响应。
    
    当工具调用次数或迭代次数达到上限时，使用已检索的数据
    生成一个尽力而为的答案，确保始终有响应输出。
    """
    # 去重收集工具结果
    seen = set()
    unique_contents = []
    for m in state["messages"]:
        if isinstance(m, ToolMessage) and m.content not in seen:
            unique_contents.append(m.content)
            seen.add(m.content)

    context_summary = state.get("context_summary", "").strip()

    # 组装上下文：压缩摘要 + 当前检索数据
    context_parts = []
    if context_summary:
        context_parts.append(f"## 压缩的研究上下文（来自先前迭代）\n\n{context_summary}")
    if unique_contents:
        context_parts.append(
            "## 检索到的数据（当前迭代）\n\n" +
            "\n\n".join(f"--- 数据源 {i} ---\n{content}" for i, content in enumerate(unique_contents, 1))
        )

    context_text = "\n\n".join(context_parts) if context_parts else "未从文档中检索到任何数据。"

    # 调用 LLM 生成降级答案
    prompt_content = (
        f"用户查询：{state.get('question')}\n\n"
        f"{context_text}\n\n"
        f"指令：\n仅使用上述数据提供最佳答案。"
    )
    response = llm.invoke([SystemMessage(content=get_fallback_response_prompt()), HumanMessage(content=prompt_content)])
    return {"messages": [response]}

def should_compress_context(state: AgentState) -> Command[Literal["compress_context", "orchestrator"]]:
    """条件节点：判断是否需要压缩上下文以节省 token。
    
    检查当前消息的 token 数量是否超过阈值，超过则进入压缩流程，
    否则直接回到编排节点继续检索。
    """
    messages = state["messages"]

    # 从最新消息中提取已执行的工具调用 ID（父块 ID 和搜索查询）
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

    # 合并已有的检索键
    updated_ids = state.get("retrieval_keys", set()) | new_ids

    # 估算当前 token 数量
    current_token_messages = estimate_context_tokens(messages)
    current_token_summary = estimate_context_tokens([HumanMessage(content=state.get("context_summary", ""))])
    current_tokens = current_token_messages + current_token_summary

    # 动态计算阈值：基础阈值 + 压缩摘要的 90%
    max_allowed = BASE_TOKEN_THRESHOLD + int(current_token_summary * TOKEN_GROWTH_FACTOR)

    # 超过阈值则压缩，否则继续编排
    goto = "compress_context" if current_tokens > max_allowed else "orchestrator"
    return Command(update={"retrieval_keys": updated_ids}, goto=goto)

def compress_context(state: AgentState, llm):
    """压缩对话上下文以节省 token。
    
    将冗长的对话历史压缩为聚焦于查询的结构化摘要，
    同时记录已执行的检索操作以避免重复。
    """
    messages = state["messages"]
    existing_summary = state.get("context_summary", "").strip()

    if not messages:
        return {}

    # 构建待压缩的对话文本
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

    # 调用 LLM 生成压缩摘要
    summary_response = llm.invoke([SystemMessage(content=get_context_compression_prompt()), HumanMessage(content=conversation_text)])
    new_summary = summary_response.content

    # 附加已执行的检索操作记录，避免智能体重复检索
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

    # 返回压缩后的摘要，并移除原始消息（仅保留系统消息）
    return {"context_summary": new_summary, "messages": [RemoveMessage(id=m.id) for m in messages[1:]]}

def collect_answer(state: AgentState):
    """收集智能体的最终答案。
    
    从最后一条助手消息中提取答案，打包为带索引的结构化结果，
    供主图的聚合节点使用。
    """
    last_message = state["messages"][-1]
    # 验证答案有效性：必须是助手消息，有内容，且没有工具调用
    is_valid = isinstance(last_message, AIMessage) and last_message.content and not last_message.tool_calls
    answer = last_message.content if is_valid else "无法生成答案。"
    return {
        "final_answer": answer,
        "agent_answers": [{"index": state["question_index"], "question": state["question"], "answer": answer}]
    }


# ========== 主图节点 ==========

def aggregate_answers(state: State, llm):
    """将多个智能体的答案聚合成一个连贯的最终响应。
    
    接收并行执行的多个智能体子图返回的答案，
    按索引排序后融合为单一的自然语言响应。
    """
    if not state.get("agent_answers"):
        return {"messages": [AIMessage(content="没有生成答案。")]}

    # 按索引排序答案
    sorted_answers = sorted(state["agent_answers"], key=lambda x: x["index"])

    # 格式化答案文本
    formatted_answers = ""
    for i, ans in enumerate(sorted_answers, start=1):
        formatted_answers += (f"\n答案 {i}:\n"f"{ans['answer']}\n")

    # 构建聚合提示，调用 LLM 生成最终响应
    user_message = HumanMessage(content=f"""原始用户问题：{state["originalQuery"]}\n检索到的答案：{formatted_answers}""")
    synthesis_response = llm.invoke([SystemMessage(content=get_aggregation_prompt()), user_message])
    return {"messages": [AIMessage(content=synthesis_response.content)]}