import os
import sys
import uuid
import streamlit as st

from agents.graph import build_graph

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

st.set_page_config(page_title="Movie Mate", page_icon="🎬", layout="wide")
st.title("🎬 Movie Mate")
st.caption("김다희의 멀티 에이전트 영화 추천 서비스")

graph = build_graph()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

with st.sidebar:
    st.header("서비스 소개")
    st.write("LangGraph + Tool Calling + TMDb + RAG 기반 영화 에이전트")
    st.write(f"현재 대화 스레드: `{st.session_state.thread_id}`")

    if st.button("새 대화 시작"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    st.subheader("예시 질문")
    st.write("- 인터스텔라 같은 감성적인 SF 영화 추천해줘")
    st.write("- 기생충 감독이 누구야?")
    st.write("- 라라랜드와 위플래시 비교해줘")
    st.write("- 인셉션 반응 요약해줘")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("debug"):
            with st.expander("디버그 정보"):
                st.json(msg["debug"])

user_input = st.chat_input("영화에 대해 질문해보세요")

if user_input:
    cleaned = user_input.strip()
    if not cleaned:
        st.warning("질문을 입력해주세요.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": cleaned})
    with st.chat_message("user"):
        st.markdown(cleaned)

    with st.chat_message("assistant"):
        with st.spinner("에이전트가 답변을 준비하고 있습니다..."):
            try:
                result = graph.invoke(
                    {"question": cleaned},
                    config={"configurable": {"thread_id": st.session_state.thread_id}},
                )

                answer = result.get("final_answer", "응답 생성에 실패했습니다.")
                debug = {
                    "intent": result.get("intent"),
                    "intent_reason": result.get("intent_reason"),
                    "draft_answer": result.get("draft_answer"),
                    "thread_id": st.session_state.thread_id,
                }

                st.markdown(answer)
                with st.expander("디버그 정보"):
                    st.json(debug)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "debug": debug}
                )

            except Exception as e:
                error_msg = f"오류가 발생했습니다: {e}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )