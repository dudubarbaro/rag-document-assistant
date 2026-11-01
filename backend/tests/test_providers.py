from app.config import Settings
from app.providers import (
    GEMINI_BASE_URL,
    get_chat_model,
    get_embeddings,
    supports_tool_calling,
)
from app.store import collection_name


def _settings(**kwargs) -> Settings:
    return Settings(_env_file=None, **kwargs)


def test_gemini_uses_openai_compatible_endpoint():
    s = _settings(llm_provider="gemini", gemini_api_key="fake")

    chat = get_chat_model(s)
    emb = get_embeddings(s)

    assert chat.model_name == "gemini-flash-latest"
    # Gemini 3: temperatura baixa pode causar loop; usamos o padrão do modelo
    assert chat.temperature is None
    assert chat.reasoning_effort == "low"
    assert chat.request_timeout == 60
    assert chat.openai_api_base == GEMINI_BASE_URL
    assert emb.model == "gemini-embedding-001"
    assert emb.openai_api_base == GEMINI_BASE_URL
    # o Gemini não aceita texto tokenizado nem lotes com mais de 100 textos
    assert emb.check_embedding_ctx_length is False
    assert emb.chunk_size == 100
    assert supports_tool_calling(s)


def test_openai_defaults():
    s = _settings(llm_provider="openai", openai_api_key="fake")
    emb = get_embeddings(s)
    assert emb.openai_api_base is None
    assert emb.check_embedding_ctx_length is True


def test_collection_per_embedding_model():
    assert collection_name(_settings(llm_provider="offline")) == "documents_offline"
    assert (
        collection_name(_settings(llm_provider="gemini")) == "documents_gemini_gemini_embedding_001"
    )
    assert (
        collection_name(_settings(llm_provider="openai"))
        == "documents_openai_text_embedding_3_small"
    )
