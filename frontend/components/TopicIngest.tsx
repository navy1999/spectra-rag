"use client";
import { useState, useEffect, useRef, useCallback } from "react";
import { TopicStatus, GraphStatus } from "@/lib/types";

// TopicIngest lets the user build a fresh knowledge graph on demand from an arXiv
// topic: POST /api/ingest/topic starts a single-slot background job; we poll
// /api/ingest/status until it finishes, then refresh the active-graph summary.
// Once a custom graph is active, queries auto-force retrieval (backend side).
export function TopicIngest() {
  const [topic, setTopic] = useState("");
  const [status, setStatus] = useState<TopicStatus>({ state: "idle" });
  const [graph, setGraph] = useState<GraphStatus | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const loadGraph = useCallback(() => {
    fetch("/api/graph", { cache: "no-store" })
      .then((r) => r.json())
      .then((g: GraphStatus) => setGraph(g))
      .catch(() => {});
  }, []);

  useEffect(() => {
    loadGraph();
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [loadGraph]);

  const poll = useCallback(() => {
    fetch("/api/ingest/status", { cache: "no-store" })
      .then((r) => r.json())
      .then((s: TopicStatus) => {
        setStatus(s);
        if (s.state !== "running" && pollRef.current) {
          clearInterval(pollRef.current);
          pollRef.current = null;
          if (s.state === "done") loadGraph();
        }
      })
      .catch(() => {});
  }, [loadGraph]);

  const start = useCallback(() => {
    const q = topic.trim();
    if (!q || status.state === "running") return;
    setStatus({ state: "running", topic: q, stage: "queued" });
    fetch("/api/ingest/topic", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: q }),
    })
      .then((r) => r.json())
      .then((j) => {
        if (j.error) {
          setStatus({ state: "error", error: j.error });
          return;
        }
        if (pollRef.current) clearInterval(pollRef.current);
        pollRef.current = setInterval(poll, 1200);
      })
      .catch(() => setStatus({ state: "error", error: "request failed" }));
  }, [topic, status.state, poll]);

  const running = status.state === "running";

  return (
    <div className="border-t border-border px-4 py-3">
      <p className="mb-1.5 font-mono text-[10px] uppercase tracking-wider text-zinc-400">Knowledge graph</p>

      {graph && (
        <p className="mb-2 text-[11px] text-zinc-500">
          {graph.nodes} nodes · {graph.edges} edges
          {graph.custom ? (
            <span className="ml-1.5 rounded bg-violet-100 px-1 py-px font-mono text-[9px] font-semibold text-violet-700">custom</span>
          ) : (
            <span className="ml-1.5 rounded bg-faint px-1 py-px font-mono text-[9px] text-zinc-400">default</span>
          )}
        </p>
      )}

      <div className="flex items-center gap-1.5">
        <input
          value={topic}
          onChange={(e) => setTopic(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && start()}
          placeholder="arXiv topic…"
          disabled={running}
          className="min-w-0 flex-1 rounded-lg border border-border bg-panel px-2 py-1.5 text-xs text-accent outline-none placeholder:text-zinc-400 focus:border-zinc-400 disabled:opacity-50"
        />
        <button
          onClick={start}
          disabled={!topic.trim() || running}
          className="shrink-0 rounded-lg bg-accent px-2.5 py-1.5 text-xs font-medium text-white transition-colors hover:bg-zinc-800 disabled:opacity-30"
        >
          Build
        </button>
      </div>

      {status.state !== "idle" && (
        <div className="mt-2 text-[11px]">
          {running && (
            <p className="text-amber-600">
              <span className="mr-1 inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-amber-500 align-middle" />
              {status.stage ?? "working"}…
            </p>
          )}
          {status.state === "done" && (
            <p className="text-emerald-600">
              ✓ {status.papers} papers → {status.nodes} nodes
              {status.compressed ? ` · PCA ${status.index_dim}d` : ""}
            </p>
          )}
          {status.state === "error" && <p className="text-red-600">✕ {status.error}</p>}
        </div>
      )}

      <p className="mt-1.5 text-[10px] leading-snug text-zinc-400">
        Builds a graph from recent arXiv papers; queries then ground in it.
      </p>
    </div>
  );
}
