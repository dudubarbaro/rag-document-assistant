"""Banco vetorial (pgvector ou memória) + registro dos documentos enviados."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Protocol

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore

from .config import Settings


@dataclass
class DocumentRecord:
    id: str
    filename: str
    pages: int
    chunks: int
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def chunk_ids(self) -> list[str]:
        return [f"{self.id}:{i}" for i in range(self.chunks)]

    def to_dict(self) -> dict:
        return asdict(self)


class DocumentRegistry(Protocol):
    def add(self, record: DocumentRecord) -> None: ...
    def get(self, doc_id: str) -> DocumentRecord | None: ...
    def list(self) -> list[DocumentRecord]: ...
    def delete(self, doc_id: str) -> None: ...


class InMemoryRegistry:
    def __init__(self) -> None:
        self._docs: dict[str, DocumentRecord] = {}

    def add(self, record: DocumentRecord) -> None:
        self._docs[record.id] = record

    def get(self, doc_id: str) -> DocumentRecord | None:
        return self._docs.get(doc_id)

    def list(self) -> list[DocumentRecord]:
        return sorted(self._docs.values(), key=lambda d: d.created_at, reverse=True)

    def delete(self, doc_id: str) -> None:
        self._docs.pop(doc_id, None)


class PostgresRegistry:
    """Tabela simples `documents` no mesmo Postgres do pgvector."""

    def __init__(self, database_url: str) -> None:
        import psycopg

        # psycopg não entende o prefixo do SQLAlchemy
        self._dsn = database_url.replace("postgresql+psycopg://", "postgresql://")
        self._psycopg = psycopg
        with self._conn() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    pages INT NOT NULL,
                    chunks INT NOT NULL,
                    created_at TEXT NOT NULL
                )"""
            )

    def _conn(self):
        return self._psycopg.connect(self._dsn, autocommit=True)

    def add(self, record: DocumentRecord) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO documents VALUES (%s, %s, %s, %s, %s)",
                (record.id, record.filename, record.pages, record.chunks, record.created_at),
            )

    def get(self, doc_id: str) -> DocumentRecord | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id, filename, pages, chunks, created_at FROM documents WHERE id = %s",
                (doc_id,),
            ).fetchone()
        return DocumentRecord(*row) if row else None

    def list(self) -> list[DocumentRecord]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, filename, pages, chunks, created_at FROM documents "
                "ORDER BY created_at DESC"
            ).fetchall()
        return [DocumentRecord(*r) for r in rows]

    def delete(self, doc_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM documents WHERE id = %s", (doc_id,))


class LocalVectorStore(InMemoryVectorStore):
    """Store em memória (sem persistência). O score já é similaridade de cosseno."""

    def _select_relevance_score_fn(self):
        return lambda score: score


def build_vector_store(settings: Settings, embeddings: Embeddings) -> VectorStore:
    if settings.database_url:
        from langchain_postgres import PGVector

        return PGVector(
            embeddings=embeddings,
            connection=settings.database_url,
            # uma coleção por provedor: embeddings de modelos diferentes têm dimensões diferentes
            collection_name=f"documents_{settings.llm_provider}",
            use_jsonb=True,
        )
    return LocalVectorStore(embeddings)


def build_registry(settings: Settings) -> DocumentRegistry:
    if settings.database_url:
        return PostgresRegistry(settings.database_url)
    return InMemoryRegistry()
