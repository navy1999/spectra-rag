"use client";
import { ChatSession } from "@/lib/types";

interface Props {
  sessions: ChatSession[];
  activeSessionId: string | null;
  onNewChat: () => void;
  onSelectSession: (id: string) => void;
}

export function Sidebar({ sessions, activeSessionId, onNewChat, onSelectSession }: Props) {
  return (
    <aside className="flex flex-col w-64 h-full bg-sidebar border-r border-border shrink-0">
      {/* Logo */}
      <div className="px-5 py-5">
        <span className="text-xl font-semibold tracking-tight text-accent">spectra</span>
        <span className="text-xl font-light text-muted">rag</span>
      </div>

      {/* New chat */}
      <div className="px-3 pb-3">
        <button
          onClick={onNewChat}
          className="w-full flex items-center gap-2 px-3 py-2 rounded-lg bg-accent text-white text-sm font-medium hover:bg-zinc-800 transition-colors"
        >
          <span className="text-base leading-none">+</span>
          New chat
        </button>
      </div>

      <div className="px-3 pb-2">
        <p className="text-xs text-muted font-medium px-2">Recent</p>
      </div>

      {/* Session list */}
      <div className="flex-1 overflow-y-auto px-3 space-y-0.5 scrollbar-thin">
        {sessions.length === 0 && (
          <p className="text-xs text-muted px-2 py-2">No chats yet</p>
        )}
        {sessions.map((s) => (
          <button
            key={s.id}
            onClick={() => onSelectSession(s.id)}
            className={`w-full text-left px-3 py-2 rounded-lg text-sm truncate transition-colors ${
              s.id === activeSessionId
                ? "bg-white shadow-sm text-accent font-medium"
                : "text-zinc-600 hover:bg-white/60"
            }`}
          >
            {s.title}
          </button>
        ))}
      </div>

      {/* Footer */}
      <div className="px-5 py-4 border-t border-border">
        <p className="text-xs text-muted">gpt-oss-120b · OpenRouter</p>
        <p className="text-[10px] text-zinc-400 mt-0.5">free-tier LLM · pro-tier controls</p>
      </div>
    </aside>
  );
}
