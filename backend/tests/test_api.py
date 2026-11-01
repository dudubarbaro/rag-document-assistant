def test_health(client):
    body = client.get("/api/health").json()
    assert body == {
        "status": "ok",
        "provider": "offline",
        "vector_store": "memory",
        "agent_enabled": False,
    }


def test_upload_list_chat_delete_flow(client, policy_bytes):
    up = client.post(
        "/api/documents", files={"file": ("politica.md", policy_bytes, "text/markdown")}
    )
    assert up.status_code == 201
    doc = up.json()
    assert doc["chunks"] > 0

    assert [d["id"] for d in client.get("/api/documents").json()] == [doc["id"]]

    chat = client.post("/api/chat", json={"question": "Quais dias são presenciais obrigatórios?"})
    assert chat.status_code == 200
    body = chat.json()
    assert body["sources"]
    assert any("terças e quintas" in s["excerpt"] for s in body["sources"])

    assert client.delete(f"/api/documents/{doc['id']}").status_code == 204
    assert client.get("/api/documents").json() == []
    assert client.delete(f"/api/documents/{doc['id']}").status_code == 404


def test_rejects_unsupported_file(client):
    r = client.post("/api/documents", files={"file": ("a.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 422


def test_validates_empty_question(client):
    assert client.post("/api/chat", json={"question": ""}).status_code == 422


def test_agent_requires_real_llm(client):
    r = client.post("/api/agent", json={"question": "oi"})
    assert r.status_code == 503


def test_provider_errors_become_clear_messages(client, policy_bytes):
    import httpx
    import openai
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    class FailingLLM(FakeListChatModel):
        error: Exception

        def _call(self, *args, **kwargs):
            raise self.error

    def failing(status: int, cls):
        response = httpx.Response(status, request=httpx.Request("POST", "http://llm"))
        return FailingLLM(responses=[], error=cls("falhou", response=response, body=None))

    client.post("/api/documents", files={"file": ("p.md", policy_bytes, "text/markdown")})
    question = {"question": "Qual o valor do auxílio home office?"}

    client.app.state.llm = failing(404, openai.NotFoundError)
    r = client.post("/api/chat", json=question)
    assert r.status_code == 502
    assert "Modelo não encontrado" in r.json()["detail"]

    client.app.state.llm = failing(429, openai.RateLimitError)
    r = client.post("/api/chat", json=question)
    assert r.status_code == 429
    assert "Aguarde" in r.json()["detail"]
