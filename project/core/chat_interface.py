import json
import re
from langchain_core.messages import HumanMessage, AIMessageChunk, ToolMessage

# 在流式输出中静默（不显示）的节点
SILENT_NODES = {"rewrite_query"}
# 需要在折叠卡片中展示的系统节点
SYSTEM_NODES = {"summarize_history", "rewrite_query"}

SYSTEM_NODE_CONFIG = {
    "rewrite_query":     {"title": "🔍 查询分析 & 重写"},
    "summarize_history": {"title": "📋 聊天历史摘要"},
}

# --- 辅助函数 ---

def make_message(content, *, title=None, node=None):
    """构造一个供 Gradio 渲染的消息字典。"""
    msg = {"role": "assistant", "content": content}
    if title or node:
        msg["metadata"] = {k: v for k, v in {"title": title, "node": node}.items() if v}
    return msg


def find_msg_idx(messages, node):
    """在消息列表中查找指定 node 元数据的索引。"""
    return next(
        (i for i, m in enumerate(messages) if m.get("metadata", {}).get("node") == node),
        None,
    )


def parse_rewrite_json(buffer):
    """从 LLM 流式 buffer 中提取第一个 JSON 对象。"""
    match = re.search(r"\{.*\}", buffer, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except Exception:
        return None


def format_rewrite_content(buffer):
    """将查询重写节点的 JSON 输出格式化为人类可读的文本。"""
    data = parse_rewrite_json(buffer)
    if not data:
        return "⏳ 正在分析查询..."
    if data.get("is_clear"):
        lines = ["✅ **查询清晰**"]
        if data.get("questions"):
            lines += ["\n**重写后的查询:**"] + [f"- {q}" for q in data["questions"]]
    else:
        lines = ["❓ **查询不清晰**"]
        clarification = data.get("clarification_needed", "")
        if clarification and clarification.strip().lower() != "no":
            lines.append(f"\n需要澄清: *{clarification}*")
    return "\n".join(lines)

# --- 辅助函数结束 ---

class ChatInterface:

    def __init__(self, rag_system):
        self.rag_system = rag_system

    def _handle_system_node(self, chunk, node, response_messages, system_node_buffer):
        """更新（或创建）可折叠的系统节点消息，并展示澄清信息。"""
        system_node_buffer[node] = system_node_buffer.get(node, "") + chunk.content
        buffer = system_node_buffer[node]
        title  = SYSTEM_NODE_CONFIG[node]["title"]
        content = format_rewrite_content(buffer) if node == "rewrite_query" else buffer

        idx = find_msg_idx(response_messages, node)
        if idx is None:
            response_messages.append(make_message(content, title=title, node=node))
        else:
            response_messages[idx]["content"] = content

        if node == "rewrite_query":
            self._surface_clarification(buffer, response_messages)

    def _surface_clarification(self, buffer, response_messages):
        """如果查询不清晰，添加/更新一条纯文本的澄清消息。"""
        data          = parse_rewrite_json(buffer) or {}
        clarification = data.get("clarification_needed", "")
        if not data.get("is_clear") and clarification.strip().lower() not in ("", "no"):
            cidx = find_msg_idx(response_messages, "clarification")
            if cidx is None:
                response_messages.append(make_message(clarification, node="clarification"))
            else:
                response_messages[cidx]["content"] = clarification

    def _handle_tool_call(self, chunk, response_messages, active_tool_calls):
        """将新的工具调用注册为可折叠消息。"""
        for tc in chunk.tool_calls:
            if tc.get("id") and tc["id"] not in active_tool_calls:
                response_messages.append(
                    make_message(f"正在运行 `{tc['name']}`...", title=f"🛠️ {tc['name']}")
                )
                active_tool_calls[tc["id"]] = len(response_messages) - 1

    def _handle_tool_result(self, chunk, response_messages, active_tool_calls):
        """将工具执行结果填入对应的可折叠消息中。"""
        idx = active_tool_calls.get(chunk.tool_call_id)
        if idx is not None:
            preview = str(chunk.content)[:300]
            suffix  = "\n..." if len(str(chunk.content)) > 300 else ""
            response_messages[idx]["content"] = f"```\n{preview}{suffix}\n```"

    def _handle_llm_token(self, chunk, node, response_messages):
        """将流式 LLM token 追加到最后一条纯助手消息中。"""
        last = response_messages[-1] if response_messages else None
        if not (last and last.get("role") == "assistant" and "metadata" not in last):
            response_messages.append(make_message(""))
        response_messages[-1]["content"] += chunk.content

    def chat(self, message, history):
        """生成器函数，逐段产出 Gradio 聊天消息字典。"""
        if not self.rag_system.agent_graph:
            yield "⚠️ 系统未初始化！"
            return

        config        = self.rag_system.get_config()
        current_state = self.rag_system.agent_graph.get_state(config)

        try:
            if current_state.next:
                # 对话挂起 -> 仅追加新消息，复用已有状态
                self.rag_system.agent_graph.update_state(config, {"messages": [HumanMessage(content=message.strip())]})
                stream_input = None
            else:
                stream_input = {"messages": [HumanMessage(content=message.strip())]}

            response_messages  = []
            active_tool_calls  = {}
            system_node_buffer = {}

            for chunk, metadata in self.rag_system.agent_graph.stream(stream_input, config=config, stream_mode="messages"):
                node = metadata.get("langgraph_node", "")

                if node in SYSTEM_NODES and isinstance(chunk, AIMessageChunk) and chunk.content:
                    self._handle_system_node(chunk, node, response_messages, system_node_buffer)

                elif hasattr(chunk, "tool_calls") and chunk.tool_calls:
                    self._handle_tool_call(chunk, response_messages, active_tool_calls)

                elif isinstance(chunk, ToolMessage):
                    self._handle_tool_result(chunk, response_messages, active_tool_calls)

                elif isinstance(chunk, AIMessageChunk) and chunk.content and node not in SILENT_NODES:
                    self._handle_llm_token(chunk, node, response_messages)

                yield response_messages

        except Exception as e:
            yield f"❌ 错误: {str(e)}"

    def clear_session(self):
        """重置 RAG 会话并刷新可观测性缓冲区。"""
        self.rag_system.reset_thread()
        self.rag_system.observability.flush()


