import streamlit as st
from agents.graph import build_graph

st.set_page_config(page_title="Movie Mate", page_icon="🎬", layout="wide")

st.title("🎬 Movie Mate")
st.caption("김다희의 멀티 에이전트 영화 추천/비교/요약 서비스")

graph = build_graph()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("서비스 소개")
    st.write("LangGraph 기반 Multi-Agent + TMDb + RAG 영화 에이전트")
    st.subheader("예시 질문")
    st.write("- 인터스텔라 같은 감성적인 SF 영화 추천해줘")
    st.write("- 기생충 감독이 누구야?")
    st.write("- 라라랜드와 위플래시 비교해줘")
    st.write("- 인셉션에 대한 전반적인 반응을 요약해줘")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("debug"):
            with st.expander("디버그 정보"):
                st.json(msg["debug"])

user_input = st.chat_input("영화에 대해 질문해보세요")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("에이전트가 답변을 준비하고 있습니다..."):
            result = graph.invoke({"question": user_input})
            answer = result.get("final_answer", "응답 생성에 실패했습니다.")
            st.markdown(answer)

            debug = {
                "intent": result.get("intent"),
                "intent_reason": result.get("intent_reason"),
                "tmdb_context": result.get("tmdb_context"),
                "rag_context": result.get("rag_context"),
                "merged_context": result.get("merged_context"),
            }

            with st.expander("디버그 정보"):
                st.json(debug)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "debug": debug}
    )