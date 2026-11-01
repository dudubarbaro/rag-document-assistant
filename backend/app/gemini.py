"""Ajustes para usar o Gemini pela API compatível com a OpenAI.

O Gemini 3 devolve, em cada chamada de ferramenta, uma "assinatura de raciocínio"
(`extra_content.google.thought_signature`) e exige recebê-la de volta na próxima
requisição. Sem ela, responde 400. O ChatOpenAI do LangChain descarta esse campo
ao remontar o histórico, então esta classe guarda a assinatura na mensagem e a
devolve no payload.

Também remove `parameters` de ferramentas sem argumentos: o Gemini recusa
`{"type": "object", "properties": {}}`, e a especificação da OpenAI permite omitir.
"""

from typing import Any

import openai
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI

EXTRA_KEY = "extra_content_by_tool_call"


class GeminiChatModel(ChatOpenAI):
    def _create_chat_result(
        self, response: dict | openai.BaseModel, generation_info: dict | None = None
    ) -> ChatResult:
        result = super()._create_chat_result(response, generation_info)
        raw = response if isinstance(response, dict) else response.model_dump(warnings=False)
        for generation, choice in zip(result.generations, raw.get("choices", []), strict=False):
            extras = {
                call["id"]: call["extra_content"]
                for call in (choice.get("message") or {}).get("tool_calls") or []
                if call.get("extra_content")
            }
            if extras:
                generation.message.additional_kwargs[EXTRA_KEY] = extras
        return result

    def _get_request_payload(self, input_: Any, *, stop: Any = None, **kwargs: Any) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)

        # Devolve as assinaturas guardadas nas mensagens anteriores do modelo
        extras: dict[str, Any] = {}
        for message in self._convert_input(input_).to_messages():
            if isinstance(message, AIMessage):
                extras.update(message.additional_kwargs.get(EXTRA_KEY, {}))
        for message in payload.get("messages", []):
            message.pop(EXTRA_KEY, None)
            for call in message.get("tool_calls") or []:
                if call.get("id") in extras:
                    call["extra_content"] = extras[call["id"]]

        for tool in payload.get("tools", []):
            function = tool.get("function", {})
            if not (function.get("parameters") or {}).get("properties"):
                function.pop("parameters", None)
        return payload
