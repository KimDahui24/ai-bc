import os
import re
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

    def _chunk_markdown(self, text: str) -> list[Document]:
        sections = re.split(r"\n##\s+", text)
        docs = []

        for idx, sec in enumerate(sections):
            sec = sec.strip()
            if not sec:
                continue

            lines = sec.splitlines()
            title = lines[0].replace("#", "").strip()
            body = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""

            paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
            if not paragraphs:
                paragraphs = [body] if body else []

            for p_idx, paragraph in enumerate(paragraphs):
                docs.append(
                    Document(
                        page_content=f"[{title}] {paragraph}",
                        metadata={
                            "section": title,
                            "chunk_index": p_idx,
                            "source": self.data_path,
                        },
                    )
                )
        return docs

    def ingest_if_needed(self):
        existing = self.vectorstore.get()
        if existing and existing.get("ids"):
            return

        path = Path(self.data_path)
        if not path.exists():
            raise FileNotFoundError(f"RAG 데이터 파일이 없습니다: {self.data_path}")

        raw = path.read_text(encoding="utf-8")
        docs = self._chunk_markdown(raw)
        if docs:
            self.vectorstore.add_documents(docs)

    def retrieve(self, query: str, k: int = 4, section: str | None = None):
        self.ingest_if_needed()
        search_kwargs = {"k": k}
        if section:
            search_kwargs["filter"] = {"section": section}
        retriever = self.vectorstore.as_retriever(search_kwargs=search_kwargs)
        return retriever.invoke(query)