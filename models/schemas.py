from typing import Literal, List, Optional
from pydantic import BaseModel, Field


class IntentResult(BaseModel):
    intent: Literal["recommend", "info", "compare", "review"] = Field(
        description="사용자 질문 의도"
    )
    reason: str = Field(description="의도 분류 이유")


class FinalAnswer(BaseModel):
    question_type: str = Field(description="질문 유형")
    answer: str = Field(description="최종 사용자 응답")
    bullets: List[str] = Field(description="핵심 포인트")
    evidence: List[str] = Field(description="근거")
    mentioned_titles: List[str] = Field(description="언급 영화 제목 목록")
    follow_up: Optional[str] = Field(default=None, description="후속 추천 질문")