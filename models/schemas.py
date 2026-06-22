from typing import Literal, List, Optional
from pydantic import BaseModel, Field


class IntentResult(BaseModel):
    intent: Literal["recommend", "info", "compare", "review"] = Field(
        description="사용자 질문 의도"
    )
    reason: str = Field(description="의도 분류 이유")


class RetrievalPlan(BaseModel):
    search_queries: List[str] = Field(description="검색용 확장 쿼리들")
    section_filter: Optional[str] = Field(
        default=None,
        description="RAG 검색 시 우선 참고할 section 이름"
    )
    needs_tmdb: bool = Field(description="TMDb 검색 필요 여부")
    needs_rag: bool = Field(description="로컬 RAG 검색 필요 여부")
    strategy_note: str = Field(description="검색 전략 설명")


class EvidenceReport(BaseModel):
    evidence_points: List[str] = Field(description="핵심 근거")
    missing_points: List[str] = Field(description="부족하거나 불명확한 부분")
    confidence: Literal["high", "medium", "low"] = Field(description="근거 신뢰도")


class FinalAnswer(BaseModel):
    question_type: str = Field(description="질문 유형")
    answer: str = Field(description="최종 사용자 응답")
    bullets: List[str] = Field(description="핵심 포인트")
    evidence: List[str] = Field(description="근거")
    mentioned_titles: List[str] = Field(description="언급 영화 제목 목록")
    follow_up: Optional[str] = Field(default=None, description="후속 질문 제안")