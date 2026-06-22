from typing import TypedDict, Literal, Annotated
from operator import add

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage

from models.schemas import IntentResult, RetrievalPlan, EvidenceReport, FinalAnswer
from services.llm_factory import get_chat_llm
from services.tools import (
    search_tmdb_movies,
    get_tmdb_movie_detail,
    search_local_knowledge,
    search_local_knowledge_multi,
)
from agents.prompts import (
    SUPERVISOR_PROMPT,
    PLANNER_PROMPT,
    TOOL_AGENT_COMMON,
    RECOMMENDER_PROMPT,
    INFO_PROMPT,
    COMPARE_PROMPT,
    REVIEW_PROMPT,
    EVIDENCE_PROMPT,
    REVIEWER_PROMPT,
)

TOOLS = [
    search_tmdb_movies,
    get_tmdb_movie_detail,
    search_local_knowledge,
    search_local_knowledge_multi,
]


class AgentState(TypedDict, total=False):
    question: str
    intent: Literal["recommend", "info", "compare", "review"]
    intent_reason: str
    plan_queries: list[str]
    plan_section: str | None
    plan_note: str
    draft_answer: str
    evidence_points: list[str]
    missing_points: list[str]
    confidence: str
    final_answer: str
    messages: Annotated[list[BaseMessage], add]


llm = get_chat_llm()
tool_llm = llm.bind_tools(TOOLS)


def supervisor_node(state: AgentState) -> AgentState:
    structured_llm = llm.with_structured_output(IntentResult)
    result = structured_llm.invoke(
        [
            {"role": "system", "content": SUPERVISOR_PROMPT},
            {"role": "user", "content": state["question"]},
        ]
    )
    return {
        "intent": result.intent,
        "intent_reason": result.reason,
        "messages": [AIMessage(content=f"intent={result.intent} / reason={result.reason}")]
    }


def planner_node(state: AgentState) -> AgentState:
    structured_llm = llm.with_structured_output(RetrievalPlan)
    result = structured_llm.invoke(
        [
            {"role": "system", "content": PLANNER_PROMPT},
            {
                "role": "user",
                "content": f"질문: {state['question']}\nintent: {state['intent']}",
            },
        ]
    )
    return {
        "plan_queries": result.search_queries,
        "plan_section": result.section_filter,
        "plan_note": result.strategy_note,
        "messages": [
            AIMessage(
                content=f"plan_queries={result.search_queries}, section={result.section_filter}, note={result.strategy_note}"
            )
        ],
    }


def route_by_intent(state: AgentState) -> str:
    return state["intent"]


def run_tool_loop(agent_instruction: str, state: AgentState) -> str:
    queries = state.get("plan_queries", [state["question"]])
    section = state.get("plan_section")

    query_text = "\n".join([f"- {q}" for q in queries])

    messages: list[BaseMessage] = [
        HumanMessage(
            content=(
                f"{TOOL_AGENT_COMMON}\n\n"
                f"{agent_instruction}\n\n"
                f"사용자 질문: {state['question']}\n"
                f"검색 전략: {state.get('plan_note', '')}\n"
                f"확장 쿼리:\n{query_text}\n"
                f"section_filter: {section}"
            )
        )
    ]

    for _ in range(5):
        ai_msg = tool_llm.invoke(messages)
        messages.append(ai_msg)

        tool_calls = getattr(ai_msg, "tool_calls", None)
        if not tool_calls:
            return ai_msg.content if isinstance(ai_msg.content, str) else str(ai_msg.content)

        for call in tool_calls:
            tool_name = call["name"]
            tool_args = call.get("args", {})
            selected_tool = next(t for t in TOOLS if t.name == tool_name)
            tool_result = selected_tool.invoke(tool_args)
            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=call["id"],
                )
            )

    return "도구 사용 후에도 충분한 답변을 생성하지 못했습니다."


def recommender_node(state: AgentState) -> AgentState:
    draft = run_tool_loop(RECOMMENDER_PROMPT, state)
    return {"draft_answer": draft, "messages": [AIMessage(content=draft)]}


def info_node(state: AgentState) -> AgentState:
    draft = run_tool_loop(INFO_PROMPT, state)
    return {"draft_answer": draft, "messages": [AIMessage(content=draft)]}


def compare_node(state: AgentState) -> AgentState:
    draft = run_tool_loop(COMPARE_PROMPT, state)
    return {"draft_answer": draft, "messages": [AIMessage(content=draft)]}


def review_node(state: AgentState) -> AgentState:
    draft = run_tool_loop(REVIEW_PROMPT, state)
    return {"draft_answer": draft, "messages": [AIMessage(content=draft)]}


def evidence_node(state: AgentState) -> AgentState:
    structured_llm = llm.with_structured_output(EvidenceReport)
    result = structured_llm.invoke(
        [
            {"role": "system", "content": EVIDENCE_PROMPT},
            {
                "role": "user",
                "content": (
                    f"질문: {state['question']}\n"
                    f"질문 유형: {state['intent']}\n"
                    f"초안:\n{state['draft_answer']}\n"
                    f"검색 전략: {state.get('plan_note', '')}\n"
                    f"확장 쿼리: {state.get('plan_queries', [])}\n"
                    f"section_filter: {state.get('plan_section')}"
                ),
            },
        ]
    )
    return {
        "evidence_points": result.evidence_points,
        "missing_points": result.missing_points,
        "confidence": result.confidence,
        "messages": [
            AIMessage(
                content=f"confidence={result.confidence}, missing={result.missing_points}"
            )
        ],
    }


def reviewer_node(state: AgentState) -> AgentState:
    structured_llm = llm.with_structured_output(FinalAnswer)
    result = structured_llm.invoke(
        [
            {"role": "system", "content": REVIEWER_PROMPT},
            {
                "role": "user",
                "content": (
                    f"질문 유형: {state['intent']}\n"
                    f"질문: {state['question']}\n"
                    f"초안:\n{state['draft_answer']}\n"
                    f"근거 포인트: {state.get('evidence_points', [])}\n"
                    f"부족한 포인트: {state.get('missing_points', [])}\n"
                    f"신뢰도: {state.get('confidence', 'medium')}\n"
                ),
            },
        ]
    )

    final_text = f"""### 답변
{result.answer}

### 핵심 포인트
""" + "\n".join([f"- {b}" for b in result.bullets]) + """

### 근거
""" + "\n".join([f"- {e}" for e in result.evidence])

    if state.get("missing_points"):
        final_text += "\n\n### 한계 또는 추가 확인 필요 사항\n"
        final_text += "\n".join([f"- {m}" for m in state["missing_points"]])

    if result.follow_up:
        final_text += f"\n\n### 다음에 물어보면 좋은 질문\n- {result.follow_up}"

    return {"final_answer": final_text, "messages": [AIMessage(content=final_text)]}


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("planner", planner_node)
    builder.add_node("recommend_agent", recommender_node)
    builder.add_node("info_agent", info_node)
    builder.add_node("compare_agent", compare_node)
    builder.add_node("review_agent", review_node)
    builder.add_node("evidence_agent", evidence_node)
    builder.add_node("reviewer", reviewer_node)

    builder.add_edge(START, "supervisor")
    builder.add_edge("supervisor", "planner")

    builder.add_conditional_edges(
        "planner",
        route_by_intent,
        {
            "recommend": "recommend_agent",
            "info": "info_agent",
            "compare": "compare_agent",
            "review": "review_agent",
        },
    )

    builder.add_edge("recommend_agent", "evidence_agent")
    builder.add_edge("info_agent", "evidence_agent")
    builder.add_edge("compare_agent", "evidence_agent")
    builder.add_edge("review_agent", "evidence_agent")
    builder.add_edge("evidence_agent", "reviewer")
    builder.add_edge("reviewer", END)

    checkpointer = InMemorySaver()
    return builder.compile(checkpointer=checkpointer)