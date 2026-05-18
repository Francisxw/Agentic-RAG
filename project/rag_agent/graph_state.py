"""
图状态定义模块

定义了 LangGraph 主图和智能体子图的状态结构，
包括消息累积、重置和集合合并等自定义累积逻辑。
"""
from typing import List, Annotated, Set
from langgraph.graph import MessagesState
import operator


def accumulate_or_reset(existing: List[dict], new: List[dict]) -> List[dict]:
    """累积或重置答案列表

    如果新列表中包含重置标记（__reset__），则清空已有列表；
    否则将新列表追加到已有列表末尾。

    Args:
        existing: 已有的答案列表
        new: 新传入的答案列表

    Returns:
        累积或重置后的答案列表
    """
    if new and any(item.get('__reset__') for item in new):
        return []
    return existing + new


def set_union(a: Set[str], b: Set[str]) -> Set[str]:
    """集合合并函数

    将两个集合取并集，用于检索键的累积。

    Args:
        a: 已有集合
        b: 新传入集合

    Returns:
        两个集合的并集
    """
    return a | b


class State(MessagesState):
    """主图状态

    包含整个对话流程所需的状态字段，
    如对话摘要、查询清晰度、重写后的问题列表、智能体答案等。
    """
    questionIsClear: bool = False           # 查询是否清晰
    conversation_summary: str = ""          # 对话上下文摘要
    originalQuery: str = ""                 # 原始用户查询
    rewrittenQuestions: List[str] = []      # 重写后的自包含问题列表
    agent_answers: Annotated[List[dict], accumulate_or_reset] = []  # 智能体答案列表（支持累积或重置）


class AgentState(MessagesState):
    """智能体子图状态

    包含单个智能体执行查询所需的字段，
    如工具调用计数、迭代计数、检索键、最终答案等。
    """
    question: str = ""                              # 当前智能体负责的具体问题
    question_index: int = 0                         # 问题索引（用于排序聚合）
    context_summary: str = ""                       # 压缩后的上下文摘要
    retrieval_keys: Annotated[Set[str], set_union] = set()  # 已执行的检索键集合（去重累积）
    final_answer: str = ""                          # 智能体生成的最终答案
    agent_answers: List[dict] = []                  # 智能体答案列表
    tool_call_count: Annotated[int, operator.add] = 0  # 工具调用累计次数
    iteration_count: Annotated[int, operator.add] = 0  # 迭代次数累计
