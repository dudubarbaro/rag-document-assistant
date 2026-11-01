export type DocumentInfo = {
  id: string;
  filename: string;
  pages: number;
  chunks: number;
  created_at: string;
};

export type Source = {
  index: number;
  doc_id: string;
  filename: string;
  page: number;
  score: number;
  excerpt: string;
};

export type AgentStep = { tool: string; input: Record<string, unknown>; output: string | null };

export type Health = {
  status: string;
  provider: string;
  vector_store: string;
  agent_enabled: boolean;
};

export type ChatTurn = { role: "user" | "assistant"; content: string };

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const detail = typeof body.detail === "string" ? body.detail : `Erro ${res.status}`;
    throw new Error(detail);
  }
  return res.status === 204 ? (undefined as T) : res.json();
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  health: () => request<Health>("/api/health"),
  listDocuments: () => request<DocumentInfo[]>("/api/documents"),
  uploadDocument: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<DocumentInfo>("/api/documents", { method: "POST", body: form });
  },
  deleteDocument: (id: string) => request<void>(`/api/documents/${id}`, { method: "DELETE" }),
  chat: (question: string, history: ChatTurn[]) =>
    request<{ answer: string; sources: Source[] }>("/api/chat", json({ question, history })),
  agent: (question: string, history: ChatTurn[]) =>
    request<{ answer: string; steps: AgentStep[] }>("/api/agent", json({ question, history })),
};
