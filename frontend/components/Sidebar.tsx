"use client";
import { ChatSession } from "@/lib/types";
import { TopicIngest } from "./TopicIngest";

interface Props {
  sessions: ChatSession[];
  activeSessionId: string | null;
  modelLabel?: string | null;
  graphRefresh?: number;
  onNewChat: () => void;
  onSelectSession: (id: string) => void;
}

function sessionMeta(s: ChatSession): string {
  const turns = s.messages.filter((m) => m.role === "user").length;
  const lastRouted = [...s.messages].reverse().find((m) => m.routeInfo?.regime);
  const path = lastRouted?.routeInfo?.path;
  if (turns === 0) return "empty";
  return path ? `${turns} turn${turns !== 1 ? "s" : ""} · ${path}` : `${turns} turn${turns !== 1 ? "s" : ""}`;
}

export function Sidebar({ sessions, activeSessionId, modelLabel, graphRefresh, onNewChat, onSelectSession }: Props) {
  return (
    <aside className="flex h-full w-64 shrink-0 flex-col border-r border-border bg-sidebar">
      {/* Wordmark */}
      <div className="flex items-center gap-2 px-5 py-4">
        <span className="text-base font-semibold tracking-tight text-accent">spectra</span>
        <span className="text-base font-light text-muted">rag</span>
        <span className="ml-auto rounded border border-border bg-panel px-1.5 py-px font-mono text-[9px] uppercase tracking-wider text-muted">
          v0.1
        </span>
      </div>

      {/* New session */}
      <div className="px-3 pb-3">
        <button
          onClick={onNewChat}
          className="flex w-full items-center gap-2 rounded-lg bg-accent px-3 py-2 text-sm font-medium text-white transition-colors hover:bg-zinc-800"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
            <line x1="12" y1="5" x2="12" y2="19" />
            <line x1="5" y1="12" x2="19" y2="12" />
          </svg>
          New session
        </button>
      </div>

      <p className="px-5 pb-1.5 font-mono text-[10px] uppercase tracking-wider text-zinc-400">Sessions</p>

      {/* Session list */}
      <div className="flex-1 space-y-0.5 overflow-y-auto px-3 scrollbar-thin">
        {sessions.length === 0 && <p className="px-2 py-2 text-xs text-muted">No sessions yet.</p>}
        {sessions.map((s) => {
          const active = s.id === activeSessionId;
          return (
            <button
              key={s.id}
              onClick={() => onSelectSession(s.id)}
              className={`group w-full rounded-lg px-3 py-2 text-left transition-colors ${
                active ? "bg-panel shadow-sm" : "hover:bg-panel/60"
              }`}
            >
              <p className={`truncate text-sm ${active ? "font-medium text-accent" : "text-zinc-600"}`}>{s.title}</p>
              <p className="mt-0.5 truncate font-mono text-[10px] text-zinc-400">{sessionMeta(s)}</p>
            </button>
          );
        })}
      </div>

      {/* Bring-your-own corpus: build a graph from an arXiv topic */}
      <TopicIngest refresh={graphRefresh} />

      {/* Footer */}
      <div className="border-t border-border px-5 py-4">
        <p className="font-mono text-[11px] text-muted">{modelLabel ?? "model unknown"}</p>
        <p className="mt-0.5 text-[10px] text-zinc-400">free-tier model · reconstructed controls</p>
      </div>
    </aside>
  );
}
