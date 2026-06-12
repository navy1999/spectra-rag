"use client";
import { useState, useCallback, useMemo } from "react";
import { Sidebar } from "@/components/Sidebar";
import { ChatArea } from "@/components/ChatArea";
import { InputBar } from "@/components/InputBar";
import { PipelineInspector } from "@/components/PipelineInspector";
import { Message, ChatSession } from "@/lib/types";
import { sendMessage } from "@/lib/api";

function genId() { return Math.random().toString(36).slice(2); }

export default function Home() {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);

  const activeSession = sessions.find((s) => s.id === activeSessionId) ?? null;
  const messages = activeSession?.messages ?? [];

  // The inspector shows the most recent routing decision in this session.
  const lastRouteInfo = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].routeInfo) return messages[i].routeInfo!;
    }
    return null;
  }, [messages]);

  const newChat = useCallback(() => {
    const id = genId();
    setSessions((prev) => [
      ...prev,
      { id, title: "New chat", messages: [], createdAt: new Date() },
    ]);
    setActiveSessionId(id);
  }, []);

  const handleSend = useCallback(
    async (query: string) => {
      let sid = activeSessionId;
      if (!sid) {
        sid = genId();
        setSessions((prev) => [
          ...prev,
          { id: sid!, title: query.slice(0, 40), messages: [], createdAt: new Date() },
        ]);
        setActiveSessionId(sid);
      }

      const userMsg: Message = { id: genId(), role: "user", content: query };
      const asstId = genId();
      const asstMsg: Message = { id: asstId, role: "assistant", content: "", streaming: true };

      setSessions((prev) =>
        prev.map((s) =>
          s.id === sid ? { ...s, messages: [...s.messages, userMsg, asstMsg] } : s
        )
      );
      setIsStreaming(true);

      await sendMessage(query, {
        onChunk: (token) => {
          setSessions((prev) =>
            prev.map((s) =>
              s.id === sid
                ? { ...s, messages: s.messages.map((m) => m.id === asstId ? { ...m, content: m.content + token } : m) }
                : s
            )
          );
        },
        onRouteInfo: (info) => {
          setSessions((prev) =>
            prev.map((s) =>
              s.id === sid
                ? { ...s, messages: s.messages.map((m) => m.id === asstId ? { ...m, routeInfo: info } : m) }
                : s
            )
          );
        },
        onDone: () => {
          setSessions((prev) =>
            prev.map((s) =>
              s.id === sid
                ? { ...s, messages: s.messages.map((m) => m.id === asstId ? { ...m, streaming: false } : m) }
                : s
            )
          );
          setIsStreaming(false);
        },
        onError: (err) => {
          setSessions((prev) =>
            prev.map((s) =>
              s.id === sid
                ? { ...s, messages: s.messages.map((m) => m.id === asstId ? { ...m, content: `Error: ${err.message}`, streaming: false } : m) }
                : s
            )
          );
          setIsStreaming(false);
        },
      });
    },
    [activeSessionId]
  );

  return (
    <div className="flex h-screen bg-offwhite overflow-hidden">
      <Sidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        onNewChat={newChat}
        onSelectSession={setActiveSessionId}
      />
      <main className="flex flex-col flex-1 min-w-0">
        <ChatArea messages={messages} onSuggest={handleSend} />
        <InputBar onSend={handleSend} disabled={isStreaming} />
      </main>
      <PipelineInspector info={lastRouteInfo} streaming={isStreaming} />
    </div>
  );
}
