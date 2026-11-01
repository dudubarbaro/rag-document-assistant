"""Simula as regras do Gemini 3 pela API compatível com a OpenAI, sem rede."""

import json

import httpx

from app.agent import build_tools, run_agent
from app.gemini import GeminiChatModel
from app.ingestion import ingest_file
from app.providers import HashingEmbeddings
from app.store import InMemoryRegistry, LocalVectorStore


def fake_gemini(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    for tool in body.get("tools", []):
        params = tool["function"].get("parameters")
        if params is not None and not params.get("properties"):
            return httpx.Response(
                400, json=[{"error": {"message": "properties: should be non-empty"}}]
            )

    tool_results = [m for m in body["messages"] if m["role"] == "tool"]
    if not tool_results:
        call = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "calcular", "arguments": json.dumps({"expressao": "150*12"})},
            "extra_content": {"google": {"thought_signature": "SIG"}},
        }
        message = {"role": "assistant", "content": None, "tool_calls": [call]}
    else:
        sent_back = [
            c for m in body["messages"] if m["role"] == "assistant" for c in m.get("tool_calls", [])
        ]
        if (
            sent_back[0].get("extra_content", {}).get("google", {}).get("thought_signature")
            != "SIG"
        ):
            return httpx.Response(400, json=[{"error": {"message": "missing thought_signature"}}])
        message = {"role": "assistant", "content": f"São R$ {tool_results[0]['content']} por ano."}

    return httpx.Response(
        200,
        json={
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": body["model"],
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        },
    )


def test_agent_works_with_gemini_rules(policy_bytes):
    store, registry = LocalVectorStore(HashingEmbeddings()), InMemoryRegistry()
    ingest_file("politica.md", policy_bytes, store, registry, 400, 40)
    llm = GeminiChatModel(
        model="gemini-flash-latest",
        api_key="fake",
        base_url="https://gemini.test/v1beta/openai/",
        http_client=httpx.Client(transport=httpx.MockTransport(fake_gemini)),
    )

    result = run_agent("Quanto recebo por ano?", llm, build_tools(store, registry, 3, 0.1))

    assert result["answer"] == "São R$ 1800 por ano."
    assert [s["tool"] for s in result["steps"]] == ["calcular"]


def overloaded(request: httpx.Request) -> httpx.Response:
    error = {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}
    return httpx.Response(503, json=[{"error": error}])


def _gemini(handler, **kwargs) -> GeminiChatModel:
    return GeminiChatModel(
        model="gemini-test",
        api_key="fake",
        base_url="https://gemini.test/v1beta/openai/",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        **kwargs,
    )


def test_agent_uses_fallback_model_when_primary_is_overloaded(policy_bytes):
    store, registry = LocalVectorStore(HashingEmbeddings()), InMemoryRegistry()
    ingest_file("politica.md", policy_bytes, store, registry, 400, 40)

    result = run_agent(
        "Quanto recebo por ano?",
        _gemini(overloaded),
        build_tools(store, registry, 3, 0.1),
        fallback=_gemini(fake_gemini),
    )

    assert result["answer"] == "São R$ 1800 por ano."


def test_overload_without_fallback_returns_clear_503(client, policy_bytes):
    client.post("/api/documents", files={"file": ("p.md", policy_bytes, "text/markdown")})
    client.app.state.llm = _gemini(overloaded)
    client.app.state.llm_fallback = None

    r = client.post("/api/chat", json={"question": "Qual o valor do auxílio home office?"})

    assert r.status_code == 503
    assert "sobrecarregado" in r.json()["detail"]


def test_rag_uses_fallback_model_when_primary_is_overloaded(client, policy_bytes):
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    client.post("/api/documents", files={"file": ("p.md", policy_bytes, "text/markdown")})
    client.app.state.llm = _gemini(overloaded)
    client.app.state.llm_fallback = FakeListChatModel(responses=["R$ 150,00 por mês [1]."])

    r = client.post("/api/chat", json={"question": "Qual o valor do auxílio home office?"})

    assert r.status_code == 200
    assert r.json()["answer"] == "R$ 150,00 por mês [1]."
