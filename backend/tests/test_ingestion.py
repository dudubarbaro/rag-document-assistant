import pytest

from app.ingestion import (
    EmptyDocumentError,
    UnsupportedFileError,
    delete_document,
    extract_pages,
    ingest_file,
    split_pages,
)
from app.providers import HashingEmbeddings
from app.store import InMemoryRegistry, LocalVectorStore

from .conftest import make_pdf


def test_extracts_text_per_pdf_page():
    pages = extract_pages("contrato.pdf", make_pdf(["Clausula primeira", "Clausula segunda"]))
    assert len(pages) == 2
    assert "primeira" in pages[0]
    assert "segunda" in pages[1]


def test_rejects_unsupported_extension():
    with pytest.raises(UnsupportedFileError):
        extract_pages("planilha.xlsx", b"x")


def test_chunks_keep_source_and_page_metadata():
    chunks = split_pages(["a " * 600, "b " * 10], "doc1", "f.pdf", chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 2
    assert all(c.metadata["doc_id"] == "doc1" and c.metadata["source"] == "f.pdf" for c in chunks)
    assert chunks[-1].metadata["page"] == 2
    assert [c.metadata["chunk"] for c in chunks] == list(range(len(chunks)))


def test_empty_pdf_raises():
    store, registry = LocalVectorStore(HashingEmbeddings()), InMemoryRegistry()
    with pytest.raises(EmptyDocumentError):
        ingest_file("vazio.pdf", make_pdf([""]), store, registry, 500, 50)


def test_ingest_and_delete_removes_all_chunks(policy_bytes):
    store, registry = LocalVectorStore(HashingEmbeddings()), InMemoryRegistry()
    record = ingest_file("politica.md", policy_bytes, store, registry, 300, 30)

    assert record.chunks > 1
    assert len(store.get_by_ids(record.chunk_ids())) == record.chunks

    assert delete_document(record.id, store, registry) is True
    assert store.get_by_ids(record.chunk_ids()) == []
    assert registry.list() == []
    assert delete_document(record.id, store, registry) is False
