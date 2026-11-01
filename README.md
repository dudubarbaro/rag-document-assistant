# Assistente de Documentos com RAG

Aplicação que permite enviar documentos (PDF, TXT, Markdown) e conversar com eles usando **RAG (Retrieval-Augmented Generation)**: o sistema busca os trechos relevantes em um **banco vetorial (PostgreSQL + pgvector)** e o LLM responde **citando as fontes**. Inclui também um **agente com tool calling** que decide sozinho quando buscar nos documentos, listar arquivos ou fazer cálculos.

**Stack:** Python 3.12 · FastAPI · LangChain · LangGraph (agente) · PostgreSQL + pgvector · OpenAI / Azure OpenAI · React + TypeScript (Vite) · Docker Compose · GitHub Actions

![Interface: chat com citações das fontes](docs/screenshot.png)

---

## Funcionalidades

- **Upload e indexação** de PDF/TXT/MD: extração de texto por página, divisão em chunks com sobreposição, geração de embeddings e gravação no pgvector.
- **Chat com RAG**: busca por similaridade de cosseno, filtro por score mínimo de relevância, resposta com citações `[1]`, `[2]` e trechos/páginas de origem.
- **Guardrails**: se nenhum trecho for relevante, a API responde "não encontrei" **sem chamar o LLM** (evita alucinação e economiza tokens). O prompt trata o conteúdo dos documentos como dado, não instrução (mitigação de *prompt injection*).
- **Agente com ferramentas** (`buscar_documentos`, `listar_documentos`, `calcular`, `data_de_hoje`), mostrando na interface quais ferramentas foram chamadas.
- **Histórico de conversa** para perguntas de acompanhamento ("e para estagiários?").
- **Provedor plugável**: OpenAI, Azure OpenAI ou **modo offline** (roda sem chave nenhuma, para demo e CI).
- **Avaliação de retrieval** com métricas *hit rate@k* e *MRR*.
- **Testes automatizados** (pytest) e CI no GitHub Actions.

## Arquitetura

```mermaid
flowchart LR
    U[Usuário] --> W[React / Vite<br/>nginx]
    W -->|/api| A[FastAPI]

    subgraph Ingestão
      A --> X[Extração de texto<br/>pypdf]
      X --> C[Chunking<br/>RecursiveCharacterTextSplitter]
      C --> E[Embeddings]
      E --> V[(PostgreSQL<br/>+ pgvector)]
    end

    subgraph Pergunta
      A --> Q[Embedding da pergunta]
      Q --> S[Busca por similaridade<br/>top-k + score mínimo]
      V --> S
      S --> P[Prompt com contexto<br/>e regras de citação]
      P --> L[LLM<br/>OpenAI / Azure]
      L --> R[Resposta + fontes]
    end
```

### Fluxo do RAG, passo a passo

1. **Ingestão** (`app/ingestion.py`): o arquivo é lido página a página e quebrado em chunks de ~1000 caracteres com 150 de sobreposição (para não cortar uma informação ao meio). Cada chunk guarda metadados: documento, nome do arquivo e página.
2. **Indexação** (`app/store.py`): cada chunk vira um vetor (embedding) e é salvo no pgvector com um ID determinístico (`<doc_id>:<n>`), o que permite apagar um documento inteiro depois.
3. **Recuperação** (`app/rag.py`): a pergunta vira um vetor; buscamos os `TOP_K` chunks mais próximos por similaridade de cosseno e descartamos os que estão abaixo de `MIN_RELEVANCE`.
4. **Geração**: os chunks entram no prompt numerados (`[1]`, `[2]`...), e o LLM é instruído a responder só com base neles e citar as fontes.

### Modo agente

No modo agente (`app/agent.py`), usamos `create_agent` do LangChain (sobre LangGraph). O LLM recebe a lista de ferramentas e decide quais chamar e em que ordem. Exemplo: *"Quanto recebo de auxílio home office por ano?"* → chama `buscar_documentos("auxílio home office")` → encontra R$ 150/mês → chama `calcular("150 * 12")` → responde R$ 1.800. A calculadora usa um avaliador de AST seguro em vez de `eval()`.

## Como rodar

### Opção 1: Docker Compose (recomendado)

```bash
cp .env.example .env          # edite: LLM_PROVIDER e sua chave
docker compose up --build
```

- Interface: http://localhost:5173
- API + Swagger: http://localhost:8000/docs

Sem chave de API, deixe `LLM_PROVIDER=offline`: tudo funciona, mas a "resposta" é um resumo extrativo (as frases mais parecidas com a pergunta) e o modo agente fica desabilitado.

Depois de subir, envie o arquivo de exemplo `samples/politica-trabalho-remoto.md` (política fictícia de uma empresa) e pergunte, por exemplo:
- *Qual o valor do auxílio home office?*
- *Estagiário pode trabalhar remoto?*
- *(modo agente)* *Quanto recebo de auxílio por ano?*

### Opção 2: localmente (desenvolvimento)

```bash
# Banco vetorial
docker compose up -d db

# Backend
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
export DATABASE_URL=postgresql+psycopg://rag:rag@localhost:5432/rag   # sem isso, usa memória
uvicorn app.main:app --reload

# Frontend (outro terminal)
cd frontend
npm install
npm run dev
```

> Para expor a porta do Postgres localmente, adicione `ports: ["5432:5432"]` ao serviço `db` no `docker-compose.yml`.

### Configuração (`.env`)

| Variável | Descrição | Padrão |
|---|---|---|
| `LLM_PROVIDER` | `openai`, `azure` ou `offline` | `offline` |
| `OPENAI_API_KEY`, `OPENAI_CHAT_MODEL`, `OPENAI_EMBEDDING_MODEL` | OpenAI | `gpt-4o-mini`, `text-embedding-3-small` |
| `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_CHAT_DEPLOYMENT`, `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Azure OpenAI | — |
| `DATABASE_URL` | Postgres com pgvector. Vazio = banco em memória | definido no compose |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Tamanho e sobreposição dos chunks | `1000` / `150` |
| `TOP_K` | Quantos chunks enviar ao LLM | `4` |
| `MIN_RELEVANCE` | Score mínimo (0 a 1) para usar um chunk | `0.30` (`0.10` offline) |

## API

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/api/health` | Provedor, banco vetorial e se o agente está ativo |
| `GET` | `/api/documents` | Lista documentos indexados |
| `POST` | `/api/documents` | Upload (multipart, campo `file`) |
| `DELETE` | `/api/documents/{id}` | Remove o documento e todos os seus vetores |
| `POST` | `/api/chat` | RAG: `{"question": "...", "history": [...]}` → resposta + fontes |
| `POST` | `/api/agent` | Agente com tools → resposta + passos executados |

```bash
curl -F file=@samples/politica-trabalho-remoto.md http://localhost:8000/api/documents
curl -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" \
     -d '{"question": "Qual o valor do auxílio home office?"}'
```

## Testes e avaliação

```bash
cd backend
pytest -q                                   # 18 testes, rodam offline (sem chave e sem banco)
python -m eval.evaluate_retrieval --k 3     # métricas de retrieval
```

Os testes usam modelos falsos: um `FakeListChatModel` para o RAG e um modelo "roteirizado" que emite chamadas de ferramenta para testar o loop do agente de ponta a ponta.

A avaliação mede se a resposta esperada aparece entre os `k` chunks recuperados (`eval/dataset.json`). Resultado no modo offline, com chunks de 300 caracteres:

| Métrica | k=1 | k=3 |
|---|---|---|
| hit rate | 64% | 91% |
| MRR | 0.64 | 0.76 |

O erro restante é instrutivo: *"Posso colocar código da empresa no ChatGPT?"* não acha o trecho sobre *"ferramentas de IA públicas"*, porque os embeddings offline são **lexicais** (comparam palavras). Embeddings semânticos (`text-embedding-3-small`) entendem que "ChatGPT" e "IA pública" são relacionados. Rode o mesmo script com `LLM_PROVIDER=openai` para comparar.

## Estrutura

```
backend/
  app/
    config.py      # configurações via variáveis de ambiente (pydantic-settings)
    providers.py   # fábrica de LLM/embeddings: OpenAI, Azure ou offline
    store.py       # pgvector ou memória + registro de documentos
    ingestion.py   # extração, chunking e indexação
    rag.py         # recuperação, prompt e geração com citações
    agent.py       # agente com tool calling
    main.py        # rotas FastAPI
  tests/           # pytest
  eval/            # avaliação de retrieval
frontend/          # React + TypeScript (Vite), servido por nginx no Docker
samples/           # documento de exemplo (empresa fictícia)
docker-compose.yml # db (pgvector) + api + web
```

## Decisões técnicas

- **pgvector em vez de um banco vetorial dedicado**: muitas empresas já têm PostgreSQL; dá para guardar vetores, metadados e dados relacionais (tabela `documents`) no mesmo lugar, com transações e backup que o time já conhece. Para volumes muito grandes, cabe índice HNSW ou um serviço dedicado (Azure AI Search, Qdrant, Pinecone).
- **Uma coleção por provedor**: embeddings de modelos diferentes têm dimensões diferentes e não podem ser comparados entre si.
- **Score mínimo antes de chamar o LLM**: corta custo e reduz alucinação quando a pergunta não tem relação com os documentos.
- **Interfaces do LangChain** (`BaseChatModel`, `Embeddings`, `VectorStore`): trocar de provedor ou de banco vetorial é uma mudança de configuração, não de código.

## Próximos passos

- Streaming da resposta (SSE) para melhorar a percepção de latência.
- Busca híbrida (vetorial + full-text/BM25) e *reranking* dos chunks.
- OCR para PDFs escaneados.
- Autenticação e separação de documentos por usuário/tenant (filtro por metadado).
- Avaliação da resposta gerada (fidelidade ao contexto) com LLM-as-judge ou RAGAS.
- Observabilidade (LangSmith / OpenTelemetry): tokens, custo e latência por requisição.
