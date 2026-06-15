"use client";
import { RouteInfo, Centroid, Diagnostic } from "@/lib/types";

interface Props {
  info: RouteInfo | null;
  streaming: boolean;
  open: boolean;
  diagnostic?: Diagnostic | null;
}

type StageStatus = "idle" | "done" | "active";

function AlgoBadge({ id, on = true }: { id: string; on?: boolean }) {
  return (
    <span
      className={`rounded px-1 py-px font-mono text-[9px] font-semibold leading-none ${
        on ? "bg-accent text-white" : "bg-faint text-zinc-300"
      }`}
    >
      {id}
    </span>
  );
}

// PCAMap renders the 2D routing space: the learned centroids, the query's
// projected point, and a dashed line to the nearest (winning) centroid.
function PCAMap({ info }: { info: RouteInfo }) {
  const W = 256,
    H = 156,
    M = 26;
  const centroids: Centroid[] = info.centroids ?? [];
  const hasQuery = typeof info.pcaX === "number" && typeof info.pcaY === "number";

  const xs = centroids.map((c) => c.x).concat(hasQuery ? [info.pcaX!] : []);
  const ys = centroids.map((c) => c.y).concat(hasQuery ? [info.pcaY!] : []);
  if (xs.length === 0) return null;

  const pad = 0.25;
  const minX = Math.min(...xs) - pad,
    maxX = Math.max(...xs) + pad;
  const minY = Math.min(...ys) - pad,
    maxY = Math.max(...ys) + pad;
  const sx = (x: number) => M + ((x - minX) / (maxX - minX)) * (W - 2 * M);
  const sy = (y: number) => H - M - ((y - minY) / (maxY - minY)) * (H - 2 * M);
  const winner = centroids.find((c) => c.name === info.regime);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      {[0.25, 0.5, 0.75].map((f) => (
        <g key={f} stroke="#f1f1f3" strokeWidth="1">
          <line x1={M + f * (W - 2 * M)} y1={M / 2} x2={M + f * (W - 2 * M)} y2={H - M / 2} />
          <line x1={M / 2} y1={M + f * (H - 2 * M)} x2={W - M / 2} y2={M + f * (H - 2 * M)} />
        </g>
      ))}
      {hasQuery && winner && (
        <line
          x1={sx(info.pcaX!)}
          y1={sy(info.pcaY!)}
          x2={sx(winner.x)}
          y2={sy(winner.y)}
          stroke="#a1a1aa"
          strokeWidth="1.2"
          strokeDasharray="3 3"
        />
      )}
      {centroids.map((c) => {
        const isWinner = c.name === info.regime;
        const fill = c.name === "logic" ? "#0ea5e9" : c.name === "creative" ? "#8b5cf6" : "#71717a";
        return (
          <g key={c.name}>
            <circle cx={sx(c.x)} cy={sy(c.y)} r={isWinner ? 7 : 5} fill={fill} opacity={isWinner ? 0.95 : 0.4} />
            <text
              x={sx(c.x)}
              y={sy(c.y) - 10}
              textAnchor="middle"
              fontSize="9"
              fill="#52525b"
              fontWeight={isWinner ? 600 : 400}
            >
              {c.name}
            </text>
          </g>
        );
      })}
      {hasQuery && (
        <g>
          <circle cx={sx(info.pcaX!)} cy={sy(info.pcaY!)} r="8" fill="#18181b" opacity="0.1" />
          <circle cx={sx(info.pcaX!)} cy={sy(info.pcaY!)} r="3.5" fill="#18181b" />
          <text x={sx(info.pcaX!)} y={sy(info.pcaY!) + 14} textAnchor="middle" fontSize="9" fill="#18181b" fontWeight={600}>
            query
          </text>
        </g>
      )}
    </svg>
  );
}

function StatusDot({ status }: { status: StageStatus }) {
  if (status === "done") {
    return (
      <span className="flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-accent">
        <svg width="8" height="8" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="4">
          <polyline points="20 6 9 17 4 12" />
        </svg>
      </span>
    );
  }
  if (status === "active") return <span className="h-4 w-4 shrink-0 animate-pulse rounded-full border-2 border-accent bg-white" />;
  return <span className="h-4 w-4 shrink-0 rounded-full border-2 border-zinc-300 bg-white" />;
}

function KV({ k, v, tone }: { k: string; v: React.ReactNode; tone?: string }) {
  return (
    <div className="flex items-center justify-between gap-2">
      <span className="text-[10px] uppercase tracking-wider text-zinc-400">{k}</span>
      <span className={`font-mono text-[11px] tnum ${tone ?? "text-accent"}`}>{v}</span>
    </div>
  );
}

export function PipelineInspector({ info, streaming, open, diagnostic }: Props) {
  if (!open) return null;

  const has = !!info && info.regime !== undefined;
  const agentic = has && info!.path === "agentic";
  const pre: StageStatus = has ? "done" : "idle";
  const tail: StageStatus = !has ? "idle" : streaming ? "active" : "done";
  const pct = (v?: number) => (typeof v === "number" ? `${Math.round(v * 100)}%` : "—");

  const stages = [
    { title: "Embed", algo: undefined as string | undefined, status: pre, detail: has ? "query → 384-d vector" : "query → embedding vector" },
    {
      title: "Project & route",
      algo: "A1",
      status: pre,
      detail: has ? `${info!.regime} · ${pct(info!.confidence)} familiar · temp ${info!.temperature.toFixed(2)}` : "PCA distance → regime + confidence",
    },
    {
      title: "Retrieve",
      algo: "A4",
      status: pre,
      detail: has ? (agentic ? `${info!.hops} hop${info!.hops !== 1 ? "s" : ""} · ${info!.chunks ?? 0} chunks · vote-gated` : "skipped (familiar query)") : "graph BFS, gated by 3-voter ensemble",
    },
    {
      title: "Synthesize",
      algo: "A3",
      status: pre,
      detail: has ? (agentic ? `redundancy ${(info!.freqPenalty ?? 0).toFixed(2)}${(info!.freqPenalty ?? 0) >= 0.8 ? " → directive" : ""}` : "no retrieved context") : "SVD redundancy penalty → prompt",
    },
    {
      title: "Guard stream",
      algo: "A2",
      status: tail,
      detail: has ? (streaming ? "watching for entity drift" : `${info!.hallucinationInterceptions} correction${info!.hallucinationInterceptions === 1 ? "" : "s"}`) : "trie corrects entity hallucinations",
    },
    { title: "Stream", algo: undefined, status: tail, detail: has ? (streaming ? "streaming…" : `done in ${info!.latencyMs}ms`) : "SSE token stream" },
  ];

  return (
    <aside className="hidden h-full w-80 shrink-0 flex-col overflow-y-auto border-l border-border bg-sidebar scrollbar-thin lg:flex">
      <div className="flex items-center justify-between px-5 pb-3 pt-4">
        <h2 className="font-mono text-[11px] uppercase tracking-wider text-muted">Pipeline inspector</h2>
        <span
          className={`rounded border px-1.5 py-0.5 font-mono text-[10px] ${
            streaming ? "border-amber-200 bg-amber-50 text-amber-600" : has ? "border-emerald-200 bg-emerald-50 text-emerald-600" : "border-border bg-faint text-zinc-400"
          }`}
        >
          {streaming ? "running" : has ? "complete" : "idle"}
        </span>
      </div>

      {/* Diagnostics */}
      {diagnostic && (
        <div className="mx-4 mb-3 rounded-lg border px-3 py-2 text-[12px] leading-relaxed"
          style={
            diagnostic.level === "error"
              ? { borderColor: "#fecaca", background: "#fef2f2", color: "#b91c1c" }
              : { borderColor: "#fde68a", background: "#fffbeb", color: "#b45309" }
          }
        >
          <p className="font-mono text-[10px] uppercase tracking-wider opacity-70">
            {diagnostic.stage ?? "system"}
            {diagnostic.code ? ` · ${diagnostic.code}` : ""}
          </p>
          <p className="mt-1">{diagnostic.message}</p>
        </div>
      )}

      {/* Decision readout */}
      {has && (
        <div className="mx-4 mb-3 rounded-xl border border-border bg-panel p-3 shadow-sm">
          <p className="mb-2 font-mono text-[10px] uppercase tracking-wider text-zinc-400">Decision</p>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1.5">
            <KV k="path" v={info!.path} tone={agentic ? "text-amber-600" : "text-accent"} />
            <KV k="regime" v={info!.regime} tone={info!.regime === "logic" ? "text-sky-600" : "text-violet-600"} />
            <KV k="confidence" v={pct(info!.confidence)} />
            <KV k="temp" v={info!.temperature.toFixed(2)} />
            {agentic && <KV k="hops" v={info!.hops} />}
            {agentic && <KV k="chunks" v={info!.chunks ?? 0} />}
            {agentic && <KV k="redundancy" v={(info!.freqPenalty ?? 0).toFixed(2)} />}
            {info!.hallucinationInterceptions > 0 && <KV k="corrections" v={info!.hallucinationInterceptions} tone="text-amber-600" />}
            <KV k="latency" v={`${info!.latencyMs}ms`} />
          </div>
          <p className="mt-2 border-t border-border pt-2 font-mono text-[10px] leading-relaxed text-zinc-500">
            sampling sent → temp {info!.temperature.toFixed(2)}
            {agentic && (info!.freqPenalty ?? 0) > 0 ? ` · frequency_penalty ${(info!.freqPenalty ?? 0).toFixed(2)}` : ""}
            <span className="text-zinc-400"> · native param where the model supports it</span>
          </p>
        </div>
      )}

      {/* PCA routing map */}
      <div className="mx-4 mb-3 rounded-xl border border-border bg-panel p-3 shadow-sm">
        <p className="mb-1 font-mono text-[10px] uppercase tracking-wider text-zinc-400">PCA routing space</p>
        {has ? (
          <>
            <PCAMap info={info!} />
            <p className="mt-1 text-[10px] leading-relaxed text-muted">
              The query projects to 2D and routes by its nearest regime centroid; distance sets confidence and temperature.
            </p>
          </>
        ) : (
          <div className="flex h-[156px] items-center justify-center">
            <p className="px-4 text-center text-xs text-muted">Run a query to see it land in routing space.</p>
          </div>
        )}
      </div>

      {/* Stage flow */}
      <div className="px-5 py-3">
        <div className="relative">
          {stages.map((s, i) => (
            <div key={s.title} className="relative flex gap-3 pb-5 last:pb-0">
              {i < stages.length - 1 && <span className="absolute left-2 top-5 bottom-0 w-px -translate-x-1/2 bg-zinc-200" />}
              <StatusDot status={s.status as StageStatus} />
              <div className="-mt-0.5 min-w-0 flex-1">
                <div className="flex items-center gap-1.5">
                  <span className={`text-xs font-medium ${s.status === "idle" ? "text-zinc-400" : "text-accent"}`}>{s.title}</span>
                  {s.algo && <AlgoBadge id={s.algo} on={!(s.algo === "A4" && has && !agentic)} />}
                </div>
                <div className="mt-0.5 font-mono text-[11px] leading-relaxed text-muted">{s.detail}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Legend */}
      <div className="mt-auto border-t border-border px-5 py-4">
        <p className="mb-2 font-mono text-[10px] uppercase tracking-wider text-zinc-400">Reconstructed surfaces</p>
        <div className="space-y-1.5 text-[11px] text-zinc-600">
          <p className="flex items-center gap-1.5"><AlgoBadge id="A1" /> router → routing + confidence</p>
          <p className="flex items-center gap-1.5"><AlgoBadge id="A2" /> trie → <span className="font-mono">logit_bias</span></p>
          <p className="flex items-center gap-1.5"><AlgoBadge id="A3" /> svd → <span className="font-mono">frequency_penalty</span> (native or directive)</p>
          <p className="flex items-center gap-1.5"><AlgoBadge id="A4" /> vote → <span className="font-mono">logprobs</span></p>
        </div>
      </div>
    </aside>
  );
}
