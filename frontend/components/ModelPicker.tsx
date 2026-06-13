"use client";
import { ModelInfo } from "@/lib/types";

interface Props {
  models: ModelInfo[];
  value: string | null;
  disabled?: boolean;
  onChange: (id: string) => void;
}

function label(m: ModelInfo): string {
  const id = m.id.replace(/:free$/, "");
  const short = id.includes("/") ? id.slice(id.indexOf("/") + 1) : id;
  const size = m.size > 0 && m.size < 1e6 ? ` · ${m.size}B` : "";
  return short + size;
}

// Native select for robustness/accessibility; the free models are sorted
// small-first by /api/models so the picker leads with the models where the
// pipeline's effect is most visible.
export function ModelPicker({ models, value, disabled, onChange }: Props) {
  const known = value && models.some((m) => m.id === value);
  return (
    <label className="flex items-center gap-1.5">
      <span className="hidden font-mono text-[10px] uppercase tracking-wider text-zinc-400 md:inline">model</span>
      <select
        value={value ?? ""}
        disabled={disabled || models.length === 0}
        onChange={(e) => onChange(e.target.value)}
        title="Generation model (OpenRouter free tier)"
        className="max-w-[210px] truncate rounded-md border border-border bg-panel px-2 py-1 font-mono text-[11px] text-accent outline-none transition-colors hover:border-zinc-400 focus:border-zinc-400 disabled:opacity-50"
      >
        {value && !known && <option value={value}>{value.replace(/:free$/, "")}</option>}
        {models.map((m) => (
          <option key={m.id} value={m.id}>
            {label(m)}
          </option>
        ))}
      </select>
    </label>
  );
}
