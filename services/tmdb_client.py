import os
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.themoviedb.org/3"


class TMDbClient:
    def __init__(self):
        token = os.getenv("TMDB_BEARER_TOKEN") or os.getenv("TMDB_API_KEY")
        if not token:
            raise RuntimeError("TMDB_BEARER_TOKEN 또는 TMDB_API_KEY 환경변수가 필요합니다.")
        self.token = token

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.token}",
            "accept": "application/json",
        }

    def _get(self, path: str, params: dict | None = None) -> dict:
        response = requests.get(
            f"{BASE_URL}{path}",
            headers=self._headers(),
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