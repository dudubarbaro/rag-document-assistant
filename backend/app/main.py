"""API HTTP (FastAPI)."""

import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Literal

import openai
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .agent import build_tools, run_agent
from .config import Settings, get_settings
from .ingestion import EmptyDocumentError, UnsupportedFileError, delete_document, ingest_file
from .providers import get_chat_model, get_embeddings, supports_tool_calling
from .rag import answer_question
from .store import build_registry, build_vector_store

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("rag")


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[ChatTurn] = Field(default_factory=list)


def llm_error_message(exc: openai.APIError) -> tuple[int, str]:
    """Traduz erros do provedor de IA em mensagens que ajudam a resolver o problema."""
    if isinstance(exc, openai.RateLimitError):
        return 429, (
            "Limite de requisições do provedor de IA atingido (comum no plano gratuito). "
            "Aguarde um minuto e tente de novo."
        )
    if isinstance(exc, openai.AuthenticationError | openai.PermissionDeniedError):
        return 502, "Chave de API recusada pelo provedor de IA. Confira a chave no .env."
    if isinstance(exc, openai.NotFoundError):
        return 502, (
            "Modelo não encontrado no provedor de IA. Confira o nome do modelo no .env "
            "(GEMINI_CHAT_MODEL / OPENAI_CHAT_MODEL)."
        )
    if isinstance(exc, openai.APIConnectionError):
        return 502, "Não foi possível conectar ao provedor de IA."
    detail = str(getattr(exc, "body", None) or "")[:300]
    return 502, f"Erro no provedor de IA: {exc} {detail}".strip()


def init_state(app: FastAPI, settings: Settings) -> None:
    if settings.debug_llm:
        from langchain_core.globals import set_debug

        set_debug(True)
    embeddings = get_embeddings(settings)
    app.state.settings = settings
    app.state.llm = get_chat_model(settings)
    app.state.vector_store = build_vector_store(settings, embeddings)
    app.state.registry = build_registry(settings)
    app.state.agent_enabled = supports_tool_calling(settings)
    log.info(
        "Provedor=%s | banco vetorial=%s",
        settings.llm_provider,
        "pgvector" if settings.database_url else "memória",
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        init_state(app, settings or get_settings())
        yield

    app = FastAPI(title="RAG Document Assistant", version="1.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )

    @app.exception_handler(openai.APIError)
    async def provider_error(request: Request, exc: openai.APIError):
        status, message = llm_error_message(exc)
        log.warning(
            "Erro do provedor de IA em %s: %s | %s", request.url.path, exc, getattr(exc, "body", "")
        )
        return JSONResponse(status_code=status, content={"detail": message})

    @app.get("/api/health")
    def health(request: Request):
        s: Settings = request.app.state.settings
        return {
            "status": "ok",
            "provider": s.llm_provider,
            "vector_store": "pgvector" if s.database_url else "memory",
            "agent_enabled": request.app.state.agent_enabled,
        }

    @app.get("/api/documents")
    def list_documents(request: Request):
        return [r.to_dict() for r in request.app.state.registry.list()]

    @app.post("/api/documents", status_code=201)
    async def upload_document(request: Request, file: UploadFile = File(...)):
        s: Settings = request.app.state.settings
        content = await file.read()
        if len(content) > s.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, f"Arquivo maior que {s.max_upload_mb} MB")
        try:
            # embeddings são I/O bloqueante: roda fora do event loop
            record = await run_in_threadpool(
                ingest_file,
                file.filename or "documento",
                content,
                request.app.state.vector_store,
                request.app.state.registry,
                s.chunk_size,
                s.chunk_overlap,
            )
        except (UnsupportedFileError, EmptyDocumentError) as exc:
            raise HTTPException(422, str(exc)) from exc
        log.info("Indexado %s: %d chunks", record.filename, record.chunks)
        return record.to_dict()

    @app.delete("/api/documents/{doc_id}", status_code=204)
    def remove_document(doc_id: str, request: Request):
        if not delete_document(doc_id, request.app.state.vector_store, request.app.state.registry):
            raise HTTPException(404, "Documento não encontrado")

    @app.post("/api/chat")
    def chat(body: ChatRequest, request: Request):
        s: Settings = request.app.state.settings
        result = answer_question(
            body.question,
            request.app.state.vector_store,
            request.app.state.llm,
            k=s.top_k,
            min_relevance=s.relevance_threshold,
            history=[(t.role, t.content) for t in body.history],
        )
        return {"answer": result.answer, "sources": [asdict(src) for src in result.sources]}

    @app.post("/api/agent")
    def agent(body: ChatRequest, request: Request):
        s: Settings = request.app.state.settings
        if not request.app.state.agent_enabled:
            raise HTTPException(
                503, "O modo agente precisa de um LLM real (LLM_PROVIDER=openai ou azure)."
            )
        tools = build_tools(
            request.app.state.vector_store,
            request.app.state.registry,
            s.top_k,
            s.relevance_threshold,
        )
        return run_agent(
            body.question,
            request.app.state.llm,
            tools,
            history=[(t.role, t.content) for t in body.history],
        )

    return app


app = create_app()
