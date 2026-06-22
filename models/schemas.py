from typing import Literal, List
from pydantic import BaseModel, Field


class IntentResult(BaseModel):
    intent: Literal["recommend", "info", "compare", "review"] = Field(
        description="사용자 질문 의도"
    )
    reason: str = Field(description="분류 이유")


class MovieAnswer(BaseModel):
    question_type: str
    summary: str
    bullets: List[str]
    recommended_titles: List[str]
    evidence: List[str]