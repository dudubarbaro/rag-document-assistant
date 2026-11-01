"""Agente com tool calling: o LLM decide quando buscar nos documentos ou usar outras ferramentas."""

import ast
import operator
from datetime import date

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langchain_core.vectorstores import VectorStore

from .rag import format_context, retrieve
from .store import DocumentRegistry

AGENT_PROMPT = """Você é um assistente corporativo que ajuda a consultar documentos internos.
Use as ferramentas disponíveis:
- `buscar_documentos` para qualquer pergunta sobre o conteúdo dos documentos (sempre cite [n]);
- `listar_documentos` para saber quais arquivos existem;
- `calcular` para contas (não faça contas de cabeça);
- `data_de_hoje` quando a pergunta depender da data atual.
Se as ferramentas não trouxerem a resposta, diga que não encontrou. Responda em português."""

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_eval(expression: str) -> float:
    """Avalia expressões aritméticas sem usar eval() (que executaria código arbitrário)."""

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            if isinstance(node.op, ast.Pow) and abs(_eval(node.right)) > 100:
                raise ValueError("expoente grande demais")
            return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.operand))
        raise ValueError("expressão não suportada")

    return _eval(ast.parse(expression, mode="eval").body)


def build_tools(
    vector_store: VectorStore, registry: DocumentRegistry, k: int, min_relevance: float
) -> list[BaseTool]:
    @tool
    def buscar_documentos(consulta: str) -> str:
        """Busca trechos relevantes nos documentos enviados. Use uma consulta curta e específica."""
        docs = retrieve(vector_store, consulta, k, min_relevance)
        if not docs:
            return "Nenhum trecho relevante encontrado."
        return format_context(docs)

    @tool
    def listar_documentos() -> str:
        """Lista os documentos disponíveis na base (nome, páginas)."""
        records = registry.list()
        if not records:
            return "Nenhum documento enviado ainda."
        return "\n".join(f"- {r.filename} ({r.pages} pág., id={r.id})" for r in records)

    @tool
    def calcular(expressao: str) -> str:
        """Calcula uma expressão aritmética, ex: '22 * 0.3' ou '(1500 + 200) / 12'."""
        try:
            return str(safe_eval(expressao))
        except (ValueError, SyntaxError, ZeroDivisionError) as exc:
            return f"Erro ao calcular: {exc}"

    @tool
    def data_de_hoje() -> str:
        """Retorna a data de hoje no formato AAAA-MM-DD."""
        return date.today().isoformat()

    return [buscar_documentos, listar_documentos, calcular, data_de_hoje]


def run_agent(
    question: str,
    llm: BaseChatModel,
    tools: list[BaseTool],
    history: list[tuple[str, str]] | None = None,
) -> dict:
    agent = create_agent(llm, tools, system_prompt=AGENT_PROMPT)
    messages = [
        HumanMessage(c) if role == "user" else AIMessage(c) for role, c in (history or [])[-6:]
    ]
    messages.append(HumanMessage(question))
    result = agent.invoke({"messages": messages})

    # Expõe os passos do agente (quais ferramentas chamou) para transparência na UI
    steps = []
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            for call in msg.tool_calls:
                steps.append({"tool": call["name"], "input": call["args"], "output": None})
        elif isinstance(msg, ToolMessage):
            for step in reversed(steps):
                if step["output"] is None and step["tool"] == msg.name:
                    step["output"] = str(msg.content)[:500]
                    break
    return {"answer": str(result["messages"][-1].content), "steps": steps}
