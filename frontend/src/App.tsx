import { FormEvent, useEffect, useRef, useState } from "react";
import { AgentStep, api, ChatTurn, DocumentInfo, Health, Source } from "./api";

type Message = ChatTurn & { sources?: Source[]; steps?: AgentStep[]; error?: boolean };
type Mode = "rag" | "agent";

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [docs, setDocs] = useState<DocumentInfo[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState<Mode>("rag");
  const [busy, setBusy] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const endRef = useRef<HTMLDivElement>(null);

  const refreshDocs = () => api.listDocuments().then(setDocs).catch((e) => setNotice(e.message));

  useEffect(() => {
    api.health().then(setHealth).catch(() => setNotice("API indisponível"));
    refreshDocs();
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function onUpload(files: FileList | null) {
    if (!files?.length) return;
    setUploading(true);
    setNotice(null);
    try {
      for (const file of Array.from(files)) {
        const doc = await api.uploadDocument(file);
        setNotice(`"${doc.filename}" indexado em ${doc.chunks} trechos.`);
      }
      await refreshDocs();
    } catch (e) {
      setNotice((e as Error).message);
    } finally {
      setUploading(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  }

  async function onDelete(doc: DocumentInfo) {
    if (!confirm(`Remover "${doc.filename}"?`)) return;
    await api.deleteDocument(doc.id).catch((e) => setNotice(e.message));
    refreshDocs();
  }

  async function onAsk(e: FormEvent) {
    e.preventDefault();
    const q = question.trim();
    if (!q || busy) return;
    const history: ChatTurn[] = messages
      .filter((m) => !m.error)
      .map(({ role, content }) => ({ role, content }));
    setMessages((m) => [...m, { role: "user", content: q }]);
    setQuestion("");
    setBusy(true);
    try {
      const reply =
        mode === "rag"
          ? await api.chat(q, history).then((r) => ({ content: r.answer, sources: r.sources }))
          : await api.agent(q, history).then((r) => ({ content: r.answer, steps: r.steps }));
      setMessages((m) => [...m, { role: "assistant", ...reply }]);
    } catch (err) {
      setMessages((m) => [...m, { role: "assistant", content: (err as Error).message, error: true }]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="layout">
      <aside className="sidebar">
        <h1>Assistente de Documentos</h1>
        {health && (
          <p className="meta">
            LLM: <b>{health.provider}</b> · vetores: <b>{health.vector_store}</b>
          </p>
        )}

        <label className={`upload ${uploading ? "disabled" : ""}`}>
          {uploading ? "Indexando..." : "+ Enviar PDF, TXT ou MD"}
          <input
            ref={fileInput}
            type="file"
            accept=".pdf,.txt,.md"
            multiple
            disabled={uploading}
            onChange={(e) => onUpload(e.target.files)}
          />
        </label>
        {notice && <p className="notice">{notice}</p>}

        <h2>Documentos ({docs.length})</h2>
        <ul className="docs">
          {docs.map((d) => (
            <li key={d.id}>
              <div>
                <span className="name">{d.filename}</span>
                <span className="meta">
                  {d.pages} pág. · {d.chunks} trechos
                </span>
              </div>
              <button className="ghost" onClick={() => onDelete(d)} title="Remover">
                ✕
              </button>
            </li>
          ))}
          {!docs.length && <li className="meta">Nenhum documento ainda.</li>}
        </ul>
      </aside>

      <main className="chat">
        <div className="modes">
          <button className={mode === "rag" ? "active" : ""} onClick={() => setMode("rag")}>
            RAG
          </button>
          <button
            className={mode === "agent" ? "active" : ""}
            onClick={() => setMode("agent")}
            disabled={health ? !health.agent_enabled : false}
            title={health?.agent_enabled ? "" : "Requer LLM_PROVIDER=openai ou azure"}
          >
            Agente (tools)
          </button>
        </div>

        <div className="messages">
          {!messages.length && (
            <div className="empty">
              Envie um documento e faça uma pergunta. Ex.: <i>"Qual o valor do auxílio home office?"</i>
            </div>
          )}
          {messages.map((m, i) => (
            <div key={i} className={`msg ${m.role} ${m.error ? "error" : ""}`}>
              <p>{m.content}</p>
              {!!m.sources?.length && (
                <details>
                  <summary>Fontes ({m.sources.length})</summary>
                  {m.sources.map((s) => (
                    <blockquote key={s.index}>
                      <b>
                        [{s.index}] {s.filename}, pág. {s.page}
                      </b>{" "}
                      <span className="meta">score {s.score.toFixed(2)}</span>
                      <br />
                      {s.excerpt}…
                    </blockquote>
                  ))}
                </details>
              )}
              {!!m.steps?.length && (
                <details>
                  <summary>Ferramentas usadas ({m.steps.length})</summary>
                  {m.steps.map((s, j) => (
                    <blockquote key={j}>
                      <b>{s.tool}</b>({JSON.stringify(s.input)})
                      <br />
                      <span className="meta">{s.output}</span>
                    </blockquote>
                  ))}
                </details>
              )}
            </div>
          ))}
          {busy && <div className="msg assistant meta">Pensando...</div>}
          <div ref={endRef} />
        </div>

        <form className="ask" onSubmit={onAsk}>
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Pergunte algo sobre os documentos..."
            maxLength={4000}
          />
          <button type="submit" disabled={busy || !question.trim()}>
            Enviar
          </button>
        </form>
      </main>
    </div>
  );
}
