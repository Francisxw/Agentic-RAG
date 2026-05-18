from typing import List
from pydantic import BaseModel, Field

class QueryAnalysis(BaseModel):
    """查询分析结果模型

    用于结构化解析 LLM 对查询清晰度和重写问题的分析结果。
    """
    is_clear: bool = Field(
        description="指示用户的问题是否清晰且可回答。"
    )
    questions: List[str] = Field(
        description="重写后的自包含问题列表。"
    )
    clarification_needed: str = Field(
        description="如果问题不清晰，提供解释。"
    )