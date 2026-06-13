"use client";
import { Diagnostic } from "@/lib/types";

interface Props {
  diagnostic: Diagnostic;
  onDismiss: () => void;
}

const TONE: Record<Diagnostic["level"], string> = {
  error: "border-red-200 bg-red-50 text-red-700",
  warn: "border-amber-200 bg-amber-50 text-amber-700",
  info: "border-border bg-faint text-muted",
};

// A slim, dismissible diagnostics strip. Operational signals (provider errors,
// fallback notices) live here and in the inspector, never in the answer body.
export function DiagnosticBanner({ diagnostic, onDismiss }: Props) {
  return (
    <div className={`mx-auto mb-2 flex max-w-3xl items-start gap-2 rounded-lg border px-3 py-2 text-[12px] ${TONE[diagnostic.level]}`}>
      <span className="mt-px font-mono text-[10px] uppercase tracking-wider opacity-70">
        {diagnostic.stage ?? diagnostic.level}
        {diagnostic.code ? ` ${diagnostic.code}` : ""}
      </span>
      <span className="flex-1 leading-relaxed">{diagnostic.message}</span>
      <button onClick={onDismiss} className="shrink-0 opacity-50 transition-opacity hover:opacity-100" aria-label="Dismiss">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
          <line x1="18" y1="6" x2="6" y2="18" />
          <line x1="6" y1="6" x2="18" y2="18" />
        </svg>
      </button>
    </div>
  );
}
