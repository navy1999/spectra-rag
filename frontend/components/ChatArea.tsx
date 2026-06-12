"use client";
import { useEffect, useRef } from "react";
import { Message } from "@/lib/types";
import { MessageBubble } from "./MessageBubble";

interface Props {
  messages: Message[];
  onSuggest: (query: string) => void;
}

const SUGGESTIONS = [
  { q: "What is FlashAttention?", hint: "fast path" },
  { q: "Which papers cite Attention Is All You Need?", hint: "multi-hop" },
  { q: "Compare BERT and GPT-3 pretraining", hint: "compare" },
];

const STEPS = [
  { n: "1", title: "Embed & project", text: "Your query becomes a vector, projected into 2D PCA space" },
  { n: "2", title: "Route", text: "Nearest regime centroid picks temperature; distance picks the path" },
  { n: "3", title: "Retrieve", text: "Novel queries trigger graph hops, gated by a 3-voter ensemble" },
  { n: "4", title: "Guarded stream", text: "A trie corrects entity hallucinations token-by-token" },
];

export function ChatArea({ messages, onSuggest }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-center px-6 overflow-y-auto">
        <div className="mb-3">
          <span className="text-4xl font-semibold text-accent">spectra</span>
          <span className="text-4xl font-light text-muted">rag</span>
        </div>
        <p className="text-muted text-sm max-w-md leading-relaxed">
          A RAG pipeline that rebuilds the LLM controls free-tier APIs leave out —
          watch every stage run live in the pipeline panel.
        </p>

        {/* How it works */}
        <div className="mt-8 grid grid-cols-2 lg:grid-cols-4 gap-3 max-w-2xl w-full">
          {STEPS.map((s) => (
            <div key={s.n} className="bg-white border border-border rounded-xl px-3.5 py-3 text-left shadow-sm">
              <div className="flex items-center gap-2 mb-1">
                <span className="w-5 h-5 rounded-full bg-accent text-white text-[10px] font-semibold flex items-center justify-center">
                  {s.n}
                </span>
                <span className="text-xs font-semibold text-accent">{s.title}</span>
              </div>
              <p className="text-[11px] text-muted leading-relaxed">{s.text}</p>
            </div>
          ))}
        </div>

        {/* Suggested prompts */}
        <p className="mt-8 text-[11px] font-medium text-muted uppercase tracking-wide">Try one</p>
        <div className="mt-2 flex gap-2 flex-wrap justify-center max-w-xl">
          {SUGGESTIONS.map(({ q, hint }) => (
            <button
              key={q}
              onClick={() => onSuggest(q)}
              className="group text-xs border border-border rounded-full pl-3 pr-2 py-1.5 text-zinc-600 bg-white hover:border-zinc-400 hover:shadow-sm transition-all flex items-center gap-2"
            >
              {q}
              <span className="text-[10px] text-muted bg-zinc-100 group-hover:bg-zinc-200 rounded-full px-1.5 py-px transition-colors">
                {hint}
              </span>
            </button>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto px-4 py-6 space-y-2 scrollbar-thin">
      <div className="max-w-3xl mx-auto">
        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
