from typing import TypedDict, Literal, Annotated
from operator import add

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_openai import AzureChatOpenAI

from models.schemas import IntentResult, FinalAnswer
from services.llm_factory import get_chat_llm
from services.tools import search_tmdb_movies, get_tmdb_movie_detail, search_local_knowledge
from agents.prompts import (
    SUPERVISOR_PROMPT,
    TOOL_AGENT_COMMON,
    RECOMMENDER_PROMPT,
    INFO_PROMPT,
    COMPARE_PROMPT,
    REVIEW_PROMPT,
    REVIEWER_PROMPT,
)

TOOLS = [search_tmdb_movies, get_tmdb_movie_detail, search_local_knowledge]


class AgentState(TypedDict, total=False):
    question: str
    intent: Literal["recommend", "info", "compare", "review"]
    intent_reason: str
    scratchpad: str
    draft_answer: str
    final_answer: str
    messages: Annotated[list[BaseMessage], add]


llm = get_chat_llm()
tool_llm: AzureChatOpenAI = llm.bind_tools(TOOLS)


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


def route_by_intent(state: AgentState) -> str:
    return state["intent"]


def run_tool_loop(agent_instruction: str, user_question: str) -> str:
    messages: list[BaseMessage] = [
        HumanMessage(
            content=f"{TOOL_AGENT_COMMON}\n\n{agent_instruction}\n\n사용자 질문: {user_question}"
        )
    ]

    for _ in range(4):
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
    draft = run_tool_loop(RECOMMENDER_PROMPT, state["question"])
    return {"draft_answer": draft, "messages": [AIMessage(content=draft)]}


def info_node(state: AgentState) -> AgentState:
    draft = run_tool_loop(INFO_PROMPT, state["question"])
    return {"draft_answer": draft, "messages": [AIMessage(content=draft)]}


def compare_node(state: AgentState) -> AgentState:
    draft = run_tool_loop(COMPARE_PROMPT, state["question"])
    return {"draft_answer": draft, "messages": [AIMessage(content=draft)]}


def review_node(state: AgentState) -> AgentState:
    draft = run_tool_loop(REVIEW_PROMPT, state["question"])
    return {"draft_answer": draft, "messages": [AIMessage(content=draft)]}


def reviewer_node(state: AgentState) -> AgentState:
    structured_llm = llm.with_structured_output(FinalAnswer)
    result = structured_llm.invoke(
        [
            {"role": "system", "content": REVIEWER_PROMPT},
            {
                "role": "user",
                "content": f"""
질문 유형: {state['intent']}
질문: {state['question']}
초안: {state['draft_answer']}

최종 사용자 응답을 정제하세요.
""",
            },
        ]
    )

    final_text = f"""### 답변
{result.answer}

### 핵심 포인트
""" + "\n".join([f"- {b}" for b in result.bullets]) + """

### 근거
""" + "\n".join([f"- {e}" for e in result.evidence])

    if result.follow_up:
        final_text += f"\n\n### 다음에 물어보면 좋은 질문\n- {result.follow_up}"

    return {"final_answer": final_text, "messages": [AIMessage(content=final_text)]}


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("recommend_agent", recommender_node)
    builder.add_node("info_agent", info_node)
    builder.add_node("compare_agent", compare_node)
    builder.add_node("review_agent", review_node)
    builder.add_node("reviewer", reviewer_node)

    builder.add_edge(START, "supervisor")

    builder.add_conditional_edges(
        "supervisor",
        route_by_intent,
        {
            "recommend": "recommend_agent",
            "info": "info_agent",
            "compare": "compare_agent",
            "review": "review_agent",
        },
    )

    builder.add_edge("recommend_agent", "reviewer")
    builder.add_edge("info_agent", "reviewer")
    builder.add_edge("compare_agent", "reviewer")
    builder.add_edge("review_agent", "reviewer")
    builder.add_edge("reviewer", END)

    checkpointer = InMemorySaver()
    return builder.compile(checkpointer=checkpointer)