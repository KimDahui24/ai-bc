import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_chroma import Chroma

from services.llm_factory import get_embeddings

load_dotenv()


class RAGService:
    def __init__(self):
        self.persist_dir = os.getenv("CHROMA_DIR", "./chroma_db")
        self.data_path = os.getenv("RAG_DATA_PATH", "./data/movie_notes.md")
        self.embedding = get_embeddings()
        self.vectorstore = Chroma(
            collection_name="movie_mate_knowledge",
            embedding_function=self.embedding,
            persist_directory=self.persist_dir,
        )

    def ingest_if_needed(self):
        existing = self.vectorstore.get()
        if existing and existing.get("ids"):
            return

        path = Path(self.data_path)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                """# Movie Mate Local Notes

인터스텔라는 감성적인 SF, 가족 서사, 우주 탐사 키워드가 강하다.
인셉션은 복잡한 구조, 꿈, 해석의 재미, 몰입감이 강한 영화다.
기생충은 계층, 블랙코미디, 사회 풍자, 긴장감이 핵심이다.
라라랜드는 사랑, 꿈, 뮤지컬, 색감, 감성적 여운이 특징이다.
위플래시는 집착, 경쟁, 음악, 긴장감이 매우 강하다.
인사이드 아웃은 가족, 성장, 감정 교육, 힐링에 적합하다.
""",
                encoding="utf-8",
            )

        raw = path.read_text(encoding="utf-8")
        chunks = [chunk.strip() for chunk in raw.split("\n") if chunk.strip()]
        docs = [
            Document(page_content=chunk, metadata={"source": str(path)})
            for chunk in chunks
        ]
        self.vectorstore.add_documents(docs)

    def retrieve(self, query: str, k: int = 4) -> list[Document]:
        self.ingest_if_needed()
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        return retriever.invoke(query)