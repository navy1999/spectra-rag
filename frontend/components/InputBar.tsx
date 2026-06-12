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
    <div className="border-t border-border bg-offwhite px-4 py-3">
      <div className="max-w-3xl mx-auto">
        <div className="flex items-end gap-2 bg-white border border-border rounded-xl shadow-sm px-3 py-2">
          <textarea
            ref={textareaRef}
            value={text}
            onChange={(e) => { setText(e.target.value); handleInput(); }}
            onKeyDown={handleKeyDown}
            rows={1}
            placeholder="Ask anything…"
            disabled={disabled}
            className="flex-1 resize-none bg-transparent text-sm text-accent placeholder-muted outline-none leading-relaxed max-h-[200px] overflow-y-auto"
          />
          <div className="flex items-center gap-2 shrink-0 pb-0.5">
            {text && (
              <span className="text-xs text-muted tabular-nums">{tokenEstimate}t</span>
            )}
            <button
              onClick={submit}
              disabled={!text.trim() || disabled}
              className="w-8 h-8 flex items-center justify-center rounded-lg bg-accent text-white disabled:opacity-30 hover:bg-zinc-800 transition-colors"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <line x1="12" y1="19" x2="12" y2="5" /><polyline points="5 12 12 5 19 12" />
              </svg>
            </button>
          </div>
        </div>
        <p className="text-center text-xs text-muted mt-1.5">
          Shift+Enter for newline · Enter to send
        </p>
      </div>
    </div>
  );
}
