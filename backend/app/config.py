"""Configuração central da aplicação, lida de variáveis de ambiente (.env)."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Lê o .env da raiz do projeto mesmo quando o backend roda de dentro de backend/
    model_config = SettingsConfigDict(env_file=("../.env", ".env"), extra="ignore")

    # Provedor de LLM/embeddings: "openai", "azure", "gemini" ou "offline" (sem chave, para demo)
    llm_provider: Literal["openai", "azure", "gemini", "offline"] = "offline"

    # OpenAI (ou qualquer API compatível com a da OpenAI, via OPENAI_BASE_URL)
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # Google Gemini (tem plano gratuito: chave em https://aistudio.google.com/apikey)
    gemini_api_key: str | None = None
    gemini_chat_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "gemini-embedding-001"

    # Azure OpenAI
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_chat_deployment: str | None = None
    azure_openai_embedding_deployment: str | None = None

    # Banco vetorial: se vazio, usa um store em memória (bom para desenvolvimento e testes)
    database_url: str | None = None

    # Parâmetros do RAG
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 4
    # Score mínimo (0..1) para um trecho ser usado. Vazio = padrão do provedor:
    # embeddings offline (lexicais) geram scores menores que os de modelos reais.
    min_relevance: float | None = None
    temperature: float = 0.0
    max_upload_mb: int = 20
    # Quantos chunks vão em cada chamada de embeddings (o Gemini aceita no máximo 100)
    embedding_batch_size: int = 100

    # Imprime no terminal tudo o que entra e sai do LLM (ótimo para estudar o fluxo)
    debug_llm: bool = False

    @property
    def relevance_threshold(self) -> float:
        if self.min_relevance is not None:
            return self.min_relevance
        return 0.10 if self.llm_provider == "offline" else 0.30


@lru_cache
def get_settings() -> Settings:
    return Settings()
