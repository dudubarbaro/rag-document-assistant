"""Ingestão: extrai texto do arquivo, divide em chunks e indexa no banco vetorial."""

import io
import uuid
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from .store import DocumentRecord, DocumentRegistry

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


class UnsupportedFileError(ValueError):
    pass


class EmptyDocumentError(ValueError):
    pass


def extract_pages(filename: str, content: bytes) -> list[str]:
    """Retorna o texto de cada página (arquivos de texto contam como 1 página)."""
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileError(
            f"Formato '{ext}' não suportado. Use: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    if ext == ".pdf":
        reader = PdfReader(io.BytesIO(content))
        return [page.extract_text() or "" for page in reader.pages]
    return [content.decode("utf-8", errors="replace")]


def split_pages(
    pages: list[str], doc_id: str, filename: str, chunk_size: int, chunk_overlap: int
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Document] = []
    for page_number, text in enumerate(pages, start=1):
        if not text.strip():
            continue
        for piece in splitter.split_text(text):
            chunks.append(
                Document(
                    page_content=piece,
                    metadata={
                        "doc_id": doc_id,
                        "source": filename,
                        "page": page_number,
                        "chunk": len(chunks),
                    },
                )
            )
    return chunks


def ingest_file(
    filename: str,
    content: bytes,
    vector_store: VectorStore,
    registry: DocumentRegistry,
    chunk_size: int,
    chunk_overlap: int,
) -> DocumentRecord:
    pages = extract_pages(filename, content)
    doc_id = uuid.uuid4().hex[:12]
    chunks = split_pages(pages, doc_id, filename, chunk_size, chunk_overlap)
    if not chunks:
        raise EmptyDocumentError(
            "Nenhum texto extraído. Se for um PDF escaneado, ele precisa de OCR antes."
        )

    record = DocumentRecord(id=doc_id, filename=filename, pages=len(pages), chunks=len(chunks))
    # IDs determinísticos permitem apagar todos os chunks de um documento depois
    vector_store.add_documents(chunks, ids=record.chunk_ids())
    registry.add(record)
    return record


def delete_document(doc_id: str, vector_store: VectorStore, registry: DocumentRegistry) -> bool:
    record = registry.get(doc_id)
    if record is None:
        return False
    vector_store.delete(ids=record.chunk_ids())
    registry.delete(doc_id)
    return True
