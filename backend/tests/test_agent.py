from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.agent import build_tools, run_agent, safe_eval
from app.ingestion import ingest_file
from app.providers import HashingEmbeddings
from app.store import InMemoryRegistry, LocalVectorStore


class ScriptedToolModel(BaseChatModel):
    """Modelo falso que devolve respostas roteirizadas (inclusive chamadas de ferramenta)."""

    script: list[AIMessage]
    seen: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any):
        return self

    def _generate(self, messages, stop=None, **kwargs) -> ChatResult:
        self.seen.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])


@pytest.fixture
def tools(policy_bytes):
    store, registry = LocalVectorStore(HashingEmbeddings()), InMemoryRegistry()
    ingest_file("politica.md", policy_bytes, store, registry, 400, 40)
    return {t.name: t for t in build_tools(store, registry, k=3, min_relevance=0.1)}


def test_safe_eval_math_and_blocks_code():
    assert safe_eval("150 * 12") == 1800
    assert safe_eval("(1200 + 300) / 3") == 500
    with pytest.raises(ValueError):
        safe_eval("__import__('os').system('ls')")


def test_tools(tools):
    assert "R$ 150,00" in tools["buscar_documentos"].invoke({"consulta": "auxílio home office"})
    assert "politica.md" in tools["listar_documentos"].invoke({})
    assert tools["calcular"].invoke({"expressao": "150 * 12"}) == "1800"
    assert "Erro" in tools["calcular"].invoke({"expressao": "1/0"})


def test_agent_calls_search_then_calculator(tools):
    model = ScriptedToolModel(
        script=[
            AIMessage(
                "",
                tool_calls=[
                    {
                        "id": "1",
                        "name": "buscar_documentos",
                        "args": {"consulta": "auxílio home office"},
                    }
                ],
            ),
            AIMessage(
                "", tool_calls=[{"id": "2", "name": "calcular", "args": {"expressao": "150*12"}}]
            ),
            AIMessage("No ano são R$ 1.800,00 de auxílio [1]."),
        ]
    )

    result = run_agent("Quanto recebo de auxílio por ano?", model, list(tools.values()))

    assert result["answer"] == "No ano são R$ 1.800,00 de auxílio [1]."
    assert [s["tool"] for s in result["steps"]] == ["buscar_documentos", "calcular"]
    assert "R$ 150,00" in result["steps"][0]["output"]
    assert result["steps"][1]["output"] == "1800"
