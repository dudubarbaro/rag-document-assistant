"""Fábrica de modelos: troca de provedor (OpenAI / Azure / offline) sem mudar o resto do código."""

import hashlib
import math
import re
import unicodedata
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from .config import Settings

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


class HashingEmbeddings(Embeddings):
    """Embeddings lexicais (feature hashing) que rodam 100% local.

    Não captura semântica como um modelo de embeddings de verdade, mas é
    determinístico e suficiente para demonstrar o pipeline e rodar testes sem chave.
    """

    def __init__(self, dimensions: int = 512):
        self.dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dimensions
        for token in _TOKEN_RE.findall(_normalize(text)):
            if len(token) < 3:  # ignora palavras muito curtas ("de", "o", "a"...)
                continue
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            vec[h % self.dimensions] += 1.0 if (h >> 8) % 2 == 0 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class OfflineChatModel(BaseChatModel):
    """LLM 'falso' para o modo offline: responde com as frases do contexto mais
    parecidas com a pergunta (resumo extrativo). Permite rodar a aplicação inteira
    sem chave de API (demo / CI).
    """

    @property
    def _llm_type(self) -> str:
        return "offline-extractive"

    @staticmethod
    def _words(text: str) -> set[str]:
        return {w for w in _TOKEN_RE.findall(_normalize(text)) if len(w) >= 3}

    def _generate(self, messages: list[BaseMessage], stop: Any = None, **kwargs: Any) -> ChatResult:
        # o contexto recuperado e a pergunta estão na última mensagem (a do usuário)
        prompt = str(messages[-1].content) if messages else ""
        context = re.search(r"<contexto>\n(.*)\n</contexto>", prompt, re.DOTALL)
        question = prompt.rsplit("Pergunta:", 1)[-1]
        if not context:
            text = "(modo offline) Não encontrei essa informação nos documentos."
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

        # quebra o contexto em frases, lembrando de qual trecho [n] cada uma veio
        candidates: list[tuple[int, str]] = []
        for block in re.split(r"\n\n(?=\[\d+\] )", context.group(1)):
            header, _, body = block.partition("\n")
            ref = re.match(r"\[(\d+)\]", header)
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", body):
                if len(sentence.strip()) > 20 and ref:
                    candidates.append((int(ref.group(1)), sentence.strip()))

        # chunks se sobrepõem (CHUNK_OVERLAP): a mesma frase pode vir de dois trechos
        unique = list(
            {sentence: (ref, sentence) for ref, sentence in reversed(candidates)}.values()
        )
        q_words = self._words(question)

        def overlap(candidate: tuple[int, str]) -> int:
            return len(q_words & self._words(candidate[1]))

        ranked = sorted(unique, key=overlap, reverse=True)[:2]
        answer = " ".join(f"{s} [{ref}]" for ref, s in ranked)
        text = f"(modo offline, sem LLM: frases mais relevantes dos documentos) {answer}"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


def get_chat_model(settings: Settings) -> BaseChatModel:
    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_chat_model,
            api_key=settings.openai_api_key,
            temperature=settings.temperature,
        )
    if settings.llm_provider == "azure":
        from langchain_openai import AzureChatOpenAI

        return AzureChatOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            azure_deployment=settings.azure_openai_chat_deployment,
            temperature=settings.temperature,
        )
    return OfflineChatModel()


def get_embeddings(settings: Settings) -> Embeddings:
    if settings.llm_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.openai_embedding_model, api_key=settings.openai_api_key
        )
    if settings.llm_provider == "azure":
        from langchain_openai import AzureOpenAIEmbeddings

        return AzureOpenAIEmbeddings(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            azure_deployment=settings.azure_openai_embedding_deployment,
        )
    return HashingEmbeddings()


def supports_tool_calling(settings: Settings) -> bool:
    return settings.llm_provider in ("openai", "azure")
