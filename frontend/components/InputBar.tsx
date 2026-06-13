"use client";
import { useState, useRef, KeyboardEvent } from "react";

interface Props {
  onSend: (query: string) => void;
  disabled: boolean;
}

export function InputBar({ onSend, disabled }: Props) {
  const [text, setText] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const tokenEstimate = Math.ceil(text.length / 4);

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  const submit = () => {
    const q = text.trim();
    if (!q || disabled) return;
    setText("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
    onSend(q);
  };

  const handleInput = () => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 200) + "px";
  };

  return (
    <div className="border-t border-border bg-canvas px-4 py-3">
      <div className="mx-auto max-w-3xl">
        <div className="flex items-end gap-2 rounded-xl border border-border bg-panel px-3 py-2 shadow-sm transition-colors focus-within:border-zinc-400">
          <textarea
            ref={textareaRef}
            value={text}
            onChange={(e) => {
              setText(e.target.value);
              handleInput();
            }}
            onKeyDown={handleKeyDown}
            rows={1}
            placeholder="Query the knowledge graph…"
            disabled={disabled}
            className="max-h-[200px] flex-1 resize-none overflow-y-auto bg-transparent text-sm leading-relaxed text-accent outline-none placeholder:text-zinc-400"
          />
          <div className="flex shrink-0 items-center gap-2 pb-0.5">
            {text && <span className="font-mono text-[11px] tabular-nums text-zinc-400">~{tokenEstimate}t</span>}
            <button
              onClick={submit}
              disabled={!text.trim() || disabled}
              className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent text-white transition-colors hover:bg-zinc-800 disabled:opacity-30"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <line x1="12" y1="19" x2="12" y2="5" />
                <polyline points="5 12 12 5 19 12" />
              </svg>
            </button>
          </div>
        </div>
        <p className="mt-1.5 text-center font-mono text-[10px] text-zinc-400">
          enter to send · shift+enter for newline · routing and retrieval run server-side
        </p>
      </div>
    </div>
  );
}
