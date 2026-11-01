from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.ingestion import ingest_file
from app.providers import HashingEmbeddings
from app.rag import NOT_FOUND_ANSWER, answer_question, build_messages, retrieve
from app.store import InMemoryRegistry, LocalVectorStore

from .conftest import SAMPLES

SAMPLE = SAMPLES / "politica-trabalho-remoto.md"


def _store_with_policy(policy_bytes) -> LocalVectorStore:
    store = LocalVectorStore(HashingEmbeddings())
    ingest_file("politica.md", policy_bytes, store, InMemoryRegistry(), 400, 40)
    return store


def test_retrieval_finds_relevant_chunk(policy_bytes):
    store = _store_with_policy(policy_bytes)
    docs = retrieve(store, "Qual o valor do auxílio home office?", k=2, min_relevance=0.1)
    assert docs
    assert "R$ 150,00" in docs[0][0].page_content


def test_answer_uses_llm_and_returns_sources(policy_bytes):
    store = _store_with_policy(policy_bytes)
    llm = FakeListChatModel(responses=["O auxílio é de R$ 150,00 por mês [1]."])

    result = answer_question("Quanto é o auxílio home office?", store, llm, k=3, min_relevance=0.1)

    assert "150" in result.answer
    assert result.sources[0].filename == "politica.md"
    assert result.sources[0].index == 1


def test_irrelevant_question_skips_llm(policy_bytes):
    store = _store_with_policy(policy_bytes)
    llm = FakeListChatModel(responses=[])  # falharia se fosse chamado

    result = answer_question("receita de bolo de cenoura", store, llm, k=3, min_relevance=0.3)

    assert result.answer == NOT_FOUND_ANSWER
    assert result.sources == []


def test_prompt_contains_context_history_and_guardrails():
    msgs = build_messages(
        "e para estagiários?", "<contexto>X</contexto>", [("user", "oi"), ("assistant", "olá")]
    )
    assert "APENAS" in msgs[0].content
    assert "não instrução" in msgs[0].content
    assert [m.type for m in msgs] == ["system", "human", "ai", "human"]
    assert "<contexto>X</contexto>" in msgs[-1].content


def test_offline_model_answers_from_context_not_system_prompt():
    from app.providers import OfflineChatModel
    from app.rag import format_context

    store = _store_with_policy(SAMPLE.read_bytes())
    docs = retrieve(store, "auxílio home office valor", k=2, min_relevance=0.1)
    reply = OfflineChatModel().invoke(build_messages("auxílio?", format_context(docs)))
    assert "R$ 150,00" in reply.content
    assert "Não invente" not in reply.content
