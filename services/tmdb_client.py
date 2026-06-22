import os
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.themoviedb.org/3"


class TMDbClient:
    def __init__(self):
        self.api_key = os.getenv("TMDB_API_KEY")
        if not self.api_key:
            raise RuntimeError("TMDB_API_KEY 환경변수가 설정되지 않았습니다.")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "accept": "application/json",
        }

    def _get(self, path: str, params: dict | None = None) -> dict:
        response = requests.get(
            f"{BASE_URL}{path}",
            headers=self.headers,
            params=params or {},
            timeout=20,
        )
        response.raise_for_status()
        return response.json()

    def search_movie(self, query: str, language: str = "ko-KR", page: int = 1) -> list[dict]:
        data = self._get(
            "/search/movie",
            params={"query": query, "language": language, "page": page},
        )
        return data.get("results", [])

    def movie_details(self, movie_id: int, language: str = "ko-KR") -> dict:
        return self._get(f"/movie/{movie_id}", params={"language": language})

    def discover_movies(
        self,
        with_genres: str | None = None,
        sort_by: str = "popularity.desc",
        language: str = "ko-KR",
        page: int = 1,
    ) -> list[dict]:
        params = {"sort_by": sort_by, "language": language, "page": page}
        if with_genres:
            params["with_genres"] = with_genres
        data = self._get("/discover/movie", params=params)
        return data.get("results", [])

    def get_movie_candidates(self, question: str) -> list[dict]:
        """
        간단한 검색 우선 전략:
        1) 질문 전체로 search_movie
        2) 결과 부족 시 discover fallback
        """
        results = self.search_movie(question)
        if results:
            return results[:5]
        return self.discover_movies()[:5]