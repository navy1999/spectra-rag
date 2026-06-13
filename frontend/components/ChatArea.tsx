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
  { q: "Which papers cite Attention Is All You Need?", hint: "agentic · multi-hop" },
  { q: "Compare BERT and GPT-3 pretraining", hint: "agentic · compare" },
];

const SURFACES = [
  {
    algo: "A1",
    kind: "adds",
    api: "routing + confidence",
    title: "Dynamic subspace router",
    desc: "Projects the query into 2D PCA space. The nearest regime centroid sets temperature; distance to it scores confidence and picks the chat or agentic path.",
  },
  {
    algo: "A2",
    kind: "reconstructs",
    api: "logit_bias",
    title: "Trie entity guard",
    desc: "Buffers the token stream and rewrites near-miss entity names to the canonical spellings from the knowledge graph.",
  },
  {
    algo: "A3",
    kind: "reconstructs",
    api: "frequency_penalty",
    title: "SVD redundancy penalty",
    desc: "Runs SVD over the retrieved context; a dominant first singular value injects an anti-repetition directive into the prompt.",
  },
  {
    algo: "A4",
    kind: "reconstructs",
    api: "logprobs",
    title: "Letter-vote evaluator",
    desc: "Three temperature-varied personas vote A/B/C on whether the context answers the question; the majority gates each retrieval hop.",
  },
];

function EmptyState({ onSuggest }: { onSuggest: (q: string) => void }) {
  return (
    <div className="bg-grid flex-1 overflow-y-auto scrollbar-thin">
      <div className="mx-auto max-w-3xl px-6 py-12">
        {/* Hero */}
        <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-muted">LLM control plane</p>
        <h1 className="mt-2 text-2xl font-semibold tracking-tight text-accent">
          Free-tier model, reconstructed controls.
        </h1>
        <p className="mt-2 max-w-xl text-sm leading-relaxed text-muted">
          Free LLM endpoints expose a token stream and nothing else. spectra-rag rebuilds the
          missing control surfaces in a Go pipeline, grounds answers in a knowledge graph, and
          surfaces every routing and retrieval decision in the inspector.
        </p>

        {/* Reconstruction map */}
        <div className="mt-7 grid grid-cols-1 gap-2.5 sm:grid-cols-2">
          {SURFACES.map((s) => (
            <div key={s.algo} className="rounded-xl border border-border bg-panel p-3.5 shadow-sm">
              <div className="flex items-center justify-between">
                <span className="rounded bg-accent px-1 py-px font-mono text-[10px] font-semibold leading-none text-white">
                  {s.algo}
                </span>
                <span className="font-mono text-[10px] text-zinc-400">
                  {s.kind} <span className="text-accent">{s.api}</span>
                </span>
              </div>
              <p className="mt-2 text-sm font-medium text-accent">{s.title}</p>
              <p className="mt-1 text-[12px] leading-relaxed text-muted">{s.desc}</p>
            </div>
          ))}
        </div>

        {/* Suggestions */}
        <p className="mt-8 font-mono text-[11px] uppercase tracking-wider text-muted">Run a query</p>
        <div className="mt-2.5 flex flex-wrap gap-2">
          {SUGGESTIONS.map(({ q, hint }) => (
            <button
              key={q}
              onClick={() => onSuggest(q)}
              className="group flex items-center gap-2 rounded-lg border border-border bg-panel py-1.5 pl-3 pr-2 text-xs text-zinc-700 transition-colors hover:border-zinc-400"
            >
              {q}
              <span className="rounded bg-faint px-1.5 py-px font-mono text-[10px] text-muted transition-colors group-hover:bg-zinc-200">
                {hint}
              </span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

export function ChatArea({ messages, onSuggest }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  if (messages.length === 0) {
    return <EmptyState onSuggest={onSuggest} />;
  }

  return (
    <div className="flex-1 overflow-y-auto px-4 py-6 scrollbar-thin">
      <div className="mx-auto max-w-3xl">
        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
