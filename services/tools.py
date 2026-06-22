from langchain_core.tools import tool

from services.tmdb_client import TMDbClient
from services.rag_service import RAGService

tmdb = TMDbClient()
rag = RAGService()


@tool
def search_tmdb_movies(query: str) -> str:
    """영화 제목이나 키워드로 TMDb에서 영화를 검색한다."""
    results = tmdb.search_movie(query)
    if not results:
        return "검색 결과 없음"

    lines = []
    for movie in results[:5]:
        lines.append(
            f"제목: {movie.get('title')} / id: {movie.get('id')} / 개봉일: {movie.get('release_date')} / 평점: {movie.get('vote_average')}"
        )
    return "\n".join(lines)


@tool
def get_tmdb_movie_detail(movie_id: int) -> str:
    """TMDb 영화 ID로 상세 정보를 가져온다."""
    detail = tmdb.movie_details(movie_id)
    genres = ", ".join([g.get("name", "") for g in detail.get("genres", [])])
    return (
        f"제목: {detail.get('title')}\n"
        f"개봉일: {detail.get('release_date')}\n"
        f"장르: {genres}\n"
        f"평점: {detail.get('vote_average')}\n"
        f"줄거리: {detail.get('overview')}"
    )


@tool
def search_local_knowledge(query: str) -> str:
    """로컬 영화 노트(RAG)에서 관련 지식을 검색한다."""
    docs = rag.retrieve(query, k=4)
    if not docs:
        return "로컬 지식 검색 결과 없음"
    return "\n".join([f"- {doc.page_content}" for doc in docs])