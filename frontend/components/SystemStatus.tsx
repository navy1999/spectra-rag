"use client";
import { SystemStatus as Status, SpectraMode } from "@/lib/types";

const MAP: Record<SpectraMode, { dot: string; label: string; tone: string; live: boolean }> = {
  checking: { dot: "bg-zinc-300", label: "Connecting", tone: "text-muted border-border bg-faint", live: true },
  pipeline: { dot: "bg-live", label: "Spectra pipeline", tone: "text-emerald-700 border-emerald-200 bg-emerald-50", live: true },
  fallback: { dot: "bg-warn", label: "Fallback LLM", tone: "text-amber-700 border-amber-200 bg-amber-50", live: false },
  unavailable: { dot: "bg-down", label: "Backend offline", tone: "text-red-700 border-red-200 bg-red-50", live: false },
};

export function SystemStatus({ status }: { status: Status }) {
  const cfg = MAP[status.status];

  return (
    <div className="flex items-center gap-2">
      <div className={`flex items-center gap-1.5 rounded-md border px-2 py-1 text-xs font-medium ${cfg.tone}`}>
        <span className="relative flex h-2 w-2">
          {cfg.live && status.status !== "checking" && (
            <span className={`absolute inline-flex h-full w-full animate-ping rounded-full opacity-60 ${cfg.dot}`} />
          )}
          <span className={`relative inline-flex h-2 w-2 rounded-full ${cfg.dot}`} />
        </span>
        {cfg.label}
      </div>
      {status.mock && (
        <span className="hidden font-mono text-[10px] uppercase tracking-wider text-amber-600 sm:inline">mock</span>
      )}
    </div>
  );
}
