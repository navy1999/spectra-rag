"use client";
import { SystemStatus as StatusComp } from "./SystemStatus";
import { SystemStatus } from "@/lib/types";

interface Props {
  title: string;
  status: SystemStatus;
  inspectorOpen: boolean;
  onToggleInspector: () => void;
}

export function TopBar({ title, status, inspectorOpen, onToggleInspector }: Props) {
  return (
    <header className="flex h-14 shrink-0 items-center justify-between gap-4 border-b border-border bg-panel/70 px-5 backdrop-blur">
      <div className="flex min-w-0 items-center gap-2.5">
        <span className="hidden font-mono text-[11px] uppercase tracking-wider text-muted sm:inline">session</span>
        <span className="truncate text-sm font-medium text-accent">{title}</span>
      </div>
      <div className="flex items-center gap-2">
        <StatusComp status={status} />
        <button
          onClick={onToggleInspector}
          title={inspectorOpen ? "Hide pipeline inspector" : "Show pipeline inspector"}
          aria-pressed={inspectorOpen}
          className={`hidden lg:flex items-center gap-1.5 rounded-md border px-2 py-1 text-xs font-medium transition-colors ${
            inspectorOpen
              ? "border-accent/15 bg-accent text-white"
              : "border-border bg-panel text-muted hover:text-accent"
          }`}
        >
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <rect x="3" y="4" width="18" height="16" rx="2" />
            <line x1="15" y1="4" x2="15" y2="20" />
          </svg>
          Inspector
        </button>
      </div>
    </header>
  );
}
