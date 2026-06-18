"use client";
import { useState, useCallback, useMemo, useEffect } from "react";
import { Sidebar } from "@/components/Sidebar";
import { ChatArea } from "@/components/ChatArea";
import { InputBar } from "@/components/InputBar";
import { TopBar } from "@/components/TopBar";
import { PipelineInspector } from "@/components/PipelineInspector";
import { DiagnosticBanner } from "@/components/DiagnosticBanner";
import { Message, ChatSession, SystemStatus, Diagnostic, ModelInfo } from "@/lib/types";
import { sendMessage } from "@/lib/api";

function genId() {
  return crypto.randomUUID();
}

export default function Home() {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [status, setStatus] = useState<SystemStatus>({ status: "checking" });
  const [diagnostic, setDiagnostic] = useState<Diagnostic | null>(null);
  const [inspectorOpen, setInspectorOpen] = useState(true);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [forceRetrieve, setForceRetrieve] = useState(false);
  const [graphRefresh, setGraphRefresh] = useState(0);

  const activeSession = sessions.find((s) => s.id === activeSessionId) ?? null;
  const messages = activeSession?.messages ?? [];

  // At-rest status read so the badge reflects pipeline availability before the
  // first query; also seeds the model picker with the backend's default.
  useEffect(() => {
    let cancelled = false;
    fetch("/api/health", { cache: "no-store" })
      .then((r) => r.json())
      .then((j) => {
        if (cancelled) return;
        setStatus({ status: j.status ?? "unavailable", model: j.model, mock: j.mock });
        if (j.model) setSelectedModel((prev) => prev ?? j.model);
      })
      .catch(() => {
        if (!cancelled) setStatus({ status: "unavailable" });
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Load the free-model catalog for the picker.
  useEffect(() => {
    let cancelled = false;
    fetch("/api/models", { cache: "no-store" })
      .then((r) => r.json())
      .then((j) => {
        if (cancelled) return;
        const list: ModelInfo[] = j.models ?? [];
        setModels(list);
        if (list.length > 0) setSelectedModel((prev) => prev ?? list[0].id);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  const lastRouteInfo = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].routeInfo) return messages[i].routeInfo!;
    }
    return null;
  }, [messages]);

  const newChat = useCallback(() => {
    // Reuse an existing empty session instead of stacking duplicate "New
    // session" entries (the old behavior minted one on every click).
    const empty = sessions.find((s) => s.messages.length === 0);
    if (empty) {
      setActiveSessionId(empty.id);
      return;
    }
    const id = genId();
    setSessions((prev) => [{ id, title: "New session", messages: [], createdAt: new Date() }, ...prev]);
    setActiveSessionId(id);
  }, [sessions]);

  const patchMessage = useCallback((sid: string, mid: string, patch: Partial<Message>) => {
    setSessions((prev) =>
      prev.map((s) =>
        s.id === sid ? { ...s, messages: s.messages.map((m) => (m.id === mid ? { ...m, ...patch } : m)) } : s
      )
    );
  }, []);

  const handleSend = useCallback(
    async (query: string) => {
      setDiagnostic(null);

      // Resolve the target session synchronously: reuse an empty one if present,
      // otherwise create it.
      let sid = activeSessionId;
      let create = false;
      if (!sid) {
        const empty = sessions.find((s) => s.messages.length === 0);
        if (empty) sid = empty.id;
        else {
          sid = genId();
          create = true;
        }
      }
      const sessionId = sid!;

      const userMsg: Message = { id: genId(), role: "user", content: query };
      const asstId = genId();
      const asstMsg: Message = { id: asstId, role: "assistant", content: "", streaming: true };

      setSessions((prev) => {
        const base = create ? [{ id: sessionId, title: query.slice(0, 48), messages: [], createdAt: new Date() }, ...prev] : prev;
        return base.map((s) => {
          if (s.id !== sessionId) return s;
          // Title an empty/"New session" session from its first message.
          const title = s.messages.length === 0 ? query.slice(0, 48) : s.title;
          return { ...s, title, messages: [...s.messages, userMsg, asstMsg] };
        });
      });
      if (create) setActiveSessionId(sessionId);
      setIsStreaming(true);

      await sendMessage(query, selectedModel, {
        onChunk: (token) =>
          setSessions((prev) =>
            prev.map((s) =>
              s.id === sessionId
                ? { ...s, messages: s.messages.map((m) => (m.id === asstId ? { ...m, content: m.content + token } : m)) }
                : s
            )
          ),
        onRouteInfo: (info) => patchMessage(sessionId, asstId, { routeInfo: info }),
        onStatus: (mode) => setStatus((s) => ({ ...s, status: mode })),
        onDiagnostic: (d) => setDiagnostic(d),
        onDone: () => {
          patchMessage(sessionId, asstId, { streaming: false });
          setIsStreaming(false);
          // Re-read the active graph in case retrieval/ingestion changed it.
          setGraphRefresh((n) => n + 1);
        },
        onError: () => {
          patchMessage(sessionId, asstId, { streaming: false, failed: true });
          setIsStreaming(false);
        },
      }, forceRetrieve);
    },
    [activeSessionId, sessions, patchMessage, selectedModel, forceRetrieve]
  );

  const headerTitle = activeSession ? activeSession.title : "New session";

  return (
    <div className="flex h-screen overflow-hidden bg-canvas">
      <Sidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        modelLabel={status.model ?? null}
        graphRefresh={graphRefresh}
        onNewChat={newChat}
        onSelectSession={setActiveSessionId}
      />

      <main className="flex min-w-0 flex-1 flex-col">
        <TopBar
          title={headerTitle}
          status={status}
          models={models}
          selectedModel={selectedModel}
          onSelectModel={setSelectedModel}
          streaming={isStreaming}
          inspectorOpen={inspectorOpen}
          onToggleInspector={() => setInspectorOpen((v) => !v)}
        />
        <ChatArea messages={messages} onSuggest={handleSend} />
        {diagnostic && (
          <div className="px-4 pt-2">
            <DiagnosticBanner diagnostic={diagnostic} onDismiss={() => setDiagnostic(null)} />
          </div>
        )}
        <InputBar
          onSend={handleSend}
          disabled={isStreaming}
          forceRetrieve={forceRetrieve}
          onToggleForce={() => setForceRetrieve((v) => !v)}
        />
      </main>

      <PipelineInspector info={lastRouteInfo} streaming={isStreaming} open={inspectorOpen} diagnostic={diagnostic} />
    </div>
  );
}
