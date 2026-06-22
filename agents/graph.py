from typing import TypedDict, Literal, Optional

from langgraph.graph import StateGraph, START, END

from models.schemas import IntentResult, MovieAnswer
from services.llm_factory import get_chat_llm
from services.tmdb_client import TMDbClient
from services.rag_service import RAGService
from agents.prompts import (
    SUPERVISOR_PROMPT,
    RETRIEVER_SUMMARY_PROMPT,
    ANSWER_PROMPT,
    REVIEWER_PROMPT,
)


class AgentState(TypedDict, total=False):
    question: str
    intent: Literal["recommend", "info", "compare", "review"]
    intent_reason: str
    tmdb_context: str
    rag_context: str
    merged_context: str
    draft_answer: str
    final_answer: str


llm = get_chat_llm()
tmdb = TMDbClient()
rag = RAGService()


def supervisor_node(state: AgentState) -> AgentState:
    structured_llm = llm.with_structured_output(IntentResult)
    result = structured_llm.invoke(
        [
            {"role": "system", "content": SUPERVISOR_PROMPT},
            {"role": "user", "content": state["question"]},
        ]
    )
    return {"intent": result.intent, "intent_reason": result.reason}


def retrieve_tmdb_node(state: AgentState) -> AgentState:
    movies = tmdb.get_movie_candidates(state["question"])
    lines = []
    for movie in movies[:3]:
        title = movie.get("title", "")
        movie_id = movie.get("id")
        if not movie_id:
            continue
        detail = tmdb.movie_details(movie_id)
        lines.append(
            f"""제목: {detail.get('title')}
개봉일: {detail.get('release_date')}
평점: {detail.get('vote_average')}
장르: {", ".join([g.get("name", "") for g in detail.get("genres", [])])}
줄거리: {detail.get('overview')}
"""
        )
    return {"tmdb_context": "\n---\n".join(lines) if lines else "TMDb 검색 결과 없음"}


def retrieve_rag_node(state: AgentState) -> AgentState:
    docs = rag.retrieve(state["question"], k=4)
    context = "\n".join([f"- {doc.page_content}" for doc in docs]) if docs else "RAG 결과 없음"
    return {"rag_context": context}


def merge_context_node(state: AgentState) -> AgentState:
    prompt = f"""
{RETRIEVER_SUMMARY_PROMPT}

[사용자 질문]
{state['question']}

[TMDb 결과]
{state.get('tmdb_context', '')}

[로컬 RAG 결과]
{state.get('rag_context', '')}
"""
    result = llm.invoke(prompt).content
    return {"merged_context": result}


def answer_node(state: AgentState) -> AgentState:
    structured_llm = llm.with_structured_output(MovieAnswer)
    result = structured_llm.invoke(
        [
            {"role": "system", "content": ANSWER_PROMPT},
            {
                "role": "user",
                "content": f"""
질문 유형: {state['intent']}
사용자 질문: {state['question']}

context:
{state['merged_context']}
""",
            },
        ]
    )

    draft = f"""질문 유형: {result.question_type}

요약:
{result.summary}

핵심 포인트:
""" + "\n".join([f"- {b}" for b in result.bullets]) + """

추천/언급 영화:
""" + "\n".join([f"- {t}" for t in result.recommended_titles]) + """

근거:
""" + "\n".join([f"- {e}" for e in result.evidence])

    return {"draft_answer": draft}


def reviewer_node(state: AgentState) -> AgentState:
    result = llm.invoke(
        [
            {"role": "system", "content": REVIEWER_PROMPT},
            {
                "role": "user",
                "content": f"""
사용자 질문:
{state['question']}

초안:
{state['draft_answer']}
""",
            },
        ]
    ).content
    return {"final_answer": result}


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("retrieve_tmdb", retrieve_tmdb_node)
    builder.add_node("retrieve_rag", retrieve_rag_node)
    builder.add_node("merge_context", merge_context_node)
    builder.add_node("answer", answer_node)
    builder.add_node("reviewer", reviewer_node)

    builder.add_edge(START, "supervisor")
    builder.add_edge("supervisor", "retrieve_tmdb")
    builder.add_edge("retrieve_tmdb", "retrieve_rag")
    builder.add_edge("retrieve_rag", "merge_context")
    builder.add_edge("merge_context", "answer")
    builder.add_edge("answer", "reviewer")
    builder.add_edge("reviewer", END)

    return builder.compile()