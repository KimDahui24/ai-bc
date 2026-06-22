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

        for sec in sections:
            sec = sec.strip()
            if not sec:
                continue

            lines = sec.splitlines()
            title = lines[0].replace("#", "").strip()
            body = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""

            paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
            if not paragraphs and body:
                paragraphs = [body]

            for idx, paragraph in enumerate(paragraphs):
                docs.append(
                    Document(
                        page_content=f"[{title}] {paragraph}",
                        metadata={
                            "section": title,
                            "chunk_index": idx,
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

    def _keyword_overlap_score(self, query: str, text: str) -> int:
        q_tokens = set(query.lower().split())
        t_tokens = set(text.lower().split())
        return len(q_tokens.intersection(t_tokens))

    def _rerank(self, query: str, docs_with_scores: list[tuple[Document, float]]) -> list[tuple[Document, float]]:
        reranked = []
        for doc, score in docs_with_scores:
            overlap = self._keyword_overlap_score(query, doc.page_content)
            final_score = score + (overlap * 0.03)
            reranked.append((doc, final_score))
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked

    def retrieve(
        self,
        query: str,
        k: int = 4,
        section: str | None = None,
        search_type: str = "mmr",
        score_threshold: float = 0.2,
    ) -> list[Document]:
        self.ingest_if_needed()

        filter_dict = {"section": section} if section else None

        docs: list[Document] = []

        if search_type == "mmr":
            retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": k,
                    "fetch_k": max(k * 2, 8),
                    **({"filter": filter_dict} if filter_dict else {}),
                },
            )
            docs = retriever.invoke(query)
        else:
            results = self.vectorstore.similarity_search_with_relevance_scores(
                query,
                k=max(k * 2, 8),
                filter=filter_dict,
            )
            filtered = [(doc, score) for doc, score in results if score >= score_threshold]
            reranked = self._rerank(query, filtered)
            docs = [doc for doc, _ in reranked[:k]]

        if not docs and section:
            retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": k, "fetch_k": max(k * 2, 8)},
            )
            docs = retriever.invoke(query)

        return docs

    def retrieve_multi_query(
        self,
        queries: list[str],
        k: int = 5,
        section: str | None = None,
    ) -> list[Document]:
        self.ingest_if_needed()

        merged: list[Document] = []
        seen = set()

        for q in queries:
            docs = self.retrieve(q, k=k, section=section, search_type="similarity")
            for doc in docs:
                key = (doc.page_content, str(doc.metadata))
                if key not in seen:
                    seen.add(key)
                    merged.append(doc)

        return merged[:k]