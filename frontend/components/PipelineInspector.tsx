"use client";
import { RouteInfo, Centroid } from "@/lib/types";

interface Props {
  info: RouteInfo | null;
  streaming: boolean;
}

type StageStatus = "idle" | "done" | "active";

const ALGO_COLORS: Record<string, string> = {
  A1: "bg-indigo-50 text-indigo-600 border-indigo-200",
  A2: "bg-amber-50 text-amber-600 border-amber-200",
  A3: "bg-rose-50 text-rose-600 border-rose-200",
  A4: "bg-emerald-50 text-emerald-600 border-emerald-200",
};

function AlgoBadge({ id }: { id: string }) {
  return (
    <span className={`text-[10px] font-semibold border rounded px-1 py-px leading-none ${ALGO_COLORS[id]}`}>
      {id}
    </span>
  );
}

function regimeStyle(regime?: string) {
  if (regime === "logic") return "bg-sky-50 text-sky-700 border-sky-200";
  if (regime === "creative") return "bg-violet-50 text-violet-700 border-violet-200";
  return "bg-zinc-50 text-zinc-600 border-zinc-200";
}

// PCAMap renders the 2D routing space: the learned centroids, the query's
// projected point, and a dashed line to the nearest (winning) centroid.
function PCAMap({ info }: { info: RouteInfo }) {
  const W = 248, H = 150, M = 26;
  const centroids: Centroid[] = info.centroids ?? [];
  const hasQuery = typeof info.pcaX === "number" && typeof info.pcaY === "number";

  const xs = centroids.map((c) => c.x).concat(hasQuery ? [info.pcaX!] : []);
  const ys = centroids.map((c) => c.y).concat(hasQuery ? [info.pcaY!] : []);
  if (xs.length === 0) return null;

  const pad = 0.25;
  const minX = Math.min(...xs) - pad, maxX = Math.max(...xs) + pad;
  const minY = Math.min(...ys) - pad, maxY = Math.max(...ys) + pad;
  const sx = (x: number) => M + ((x - minX) / (maxX - minX)) * (W - 2 * M);
  const sy = (y: number) => H - M - ((y - minY) / (maxY - minY)) * (H - 2 * M);

  const winner = centroids.find((c) => c.name === info.regime);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      {/* grid */}
      {[0.25, 0.5, 0.75].map((f) => (
        <g key={f} stroke="#ececec" strokeWidth="1">
          <line x1={M + f * (W - 2 * M)} y1={M / 2} x2={M + f * (W - 2 * M)} y2={H - M / 2} />
          <line x1={M / 2} y1={M + f * (H - 2 * M)} x2={W - M / 2} y2={M + f * (H - 2 * M)} />
        </g>
      ))}
      {/* line query -> winning centroid */}
      {hasQuery && winner && (
        <line
          x1={sx(info.pcaX!)} y1={sy(info.pcaY!)}
          x2={sx(winner.x)} y2={sy(winner.y)}
          stroke="#a1a1aa" strokeWidth="1.2" strokeDasharray="3 3"
        />
      )}
      {/* centroids */}
      {centroids.map((c) => {
        const isWinner = c.name === info.regime;
        const fill = c.name === "logic" ? "#0ea5e9" : c.name === "creative" ? "#8b5cf6" : "#71717a";
        return (
          <g key={c.name}>
            <circle cx={sx(c.x)} cy={sy(c.y)} r={isWinner ? 7 : 5} fill={fill} opacity={isWinner ? 0.9 : 0.45} />
            <text x={sx(c.x)} y={sy(c.y) - 10} textAnchor="middle" fontSize="9" fill="#52525b" fontWeight={isWinner ? 600 : 400}>
              {c.name}
            </text>
          </g>
        );
      })}
      {/* query point */}
      {hasQuery && (
        <g>
          <circle cx={sx(info.pcaX!)} cy={sy(info.pcaY!)} r="8" fill="#18181b" opacity="0.12" />
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
      <span className="w-4 h-4 rounded-full bg-accent flex items-center justify-center shrink-0">
        <svg width="8" height="8" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="4">
          <polyline points="20 6 9 17 4 12" />
        </svg>
      </span>
    );
  }
  if (status === "active") {
    return <span className="w-4 h-4 rounded-full border-2 border-accent bg-white shrink-0 animate-pulse" />;
  }
  return <span className="w-4 h-4 rounded-full border-2 border-zinc-300 bg-white shrink-0" />;
}

interface Stage {
  title: string;
  algo?: string;
  status: StageStatus;
  detail: React.ReactNode;
}

export function PipelineInspector({ info, streaming }: Props) {
  const has = !!info && info.regime !== undefined;
  const pre: StageStatus = has ? "done" : "idle";
  const tail: StageStatus = !has ? "idle" : streaming ? "active" : "done";

  const pct = (v?: number) => (typeof v === "number" ? `${Math.round(v * 100)}%` : "—");

  const stages: Stage[] = [
    {
      title: "Embed",
      status: pre,
      detail: has ? "query → 384-d vector" : "query → embedding vector",
    },
    {
      title: "Project & route",
      algo: "A1",
      status: pre,
      detail: has ? (
        <span className="flex items-center gap-1.5 flex-wrap">
          <span className={`border rounded-full px-1.5 py-px text-[10px] font-medium ${regimeStyle(info!.regime)}`}>
            {info!.regime}
          </span>
          <span>{pct(info!.confidence)} familiar · temp {info!.temperature.toFixed(2)}</span>
        </span>
      ) : (
        "PCA distance → regime + confidence"
      ),
    },
    {
      title: "Retrieve",
      algo: "A4",
      status: pre,
      detail: has
        ? info!.path === "agentic"
          ? `${info!.hops} hop${info!.hops !== 1 ? "s" : ""} · ${info!.chunks ?? 0} chunks · vote-gated`
          : "skipped — familiar query, direct chat"
        : "graph BFS, gated by 3-voter ensemble",
    },
    {
      title: "Synthesize",
      algo: "A3",
      status: pre,
      detail: has
        ? (info!.freqPenalty ?? 0) >= 0.8
          ? `redundancy ${info!.freqPenalty!.toFixed(2)} → anti-repetition directive`
          : `redundancy ${(info!.freqPenalty ?? 0).toFixed(2)} — no directive needed`
        : "SVD redundancy penalty → prompt",
    },
    {
      title: "Guard stream",
      algo: "A2",
      status: tail,
      detail: has
        ? streaming
          ? "watching tokens for entity drift…"
          : `${info!.hallucinationInterceptions} entit${info!.hallucinationInterceptions === 1 ? "y" : "ies"} corrected`
        : "trie corrects entity hallucinations",
    },
    {
      title: "Stream",
      status: tail,
      detail: has
        ? streaming
          ? "tokens streaming…"
          : `done in ${info!.latencyMs}ms`
        : "SSE token stream to browser",
    },
  ];

  return (
    <aside className="hidden xl:flex flex-col w-80 h-full bg-offwhite border-l border-border shrink-0 overflow-y-auto scrollbar-thin">
      <div className="px-5 pt-5 pb-3 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-accent">Pipeline</h2>
        <span
          className={`text-[10px] font-medium rounded-full px-2 py-0.5 border ${
            streaming
              ? "bg-amber-50 text-amber-600 border-amber-200"
              : has
                ? "bg-emerald-50 text-emerald-600 border-emerald-200"
                : "bg-zinc-100 text-zinc-500 border-zinc-200"
          }`}
        >
          {streaming ? "running" : has ? "complete" : "idle"}
        </span>
      </div>

      {/* PCA routing map */}
      <div className="mx-4 bg-white border border-border rounded-xl p-3 shadow-sm">
        <p className="text-[10px] font-medium text-muted uppercase tracking-wide mb-1">
          PCA routing space
        </p>
        {has ? (
          <>
            <PCAMap info={info!} />
            <p className="text-[10px] text-muted leading-relaxed mt-1">
              The query embeds, projects to 2D, and routes by its nearest regime
              centroid — distance sets confidence and temperature.
            </p>
          </>
        ) : (
          <div className="h-[150px] flex items-center justify-center">
            <p className="text-xs text-muted text-center px-4">
              Send a query to see it land in routing space
            </p>
          </div>
        )}
      </div>

      {/* Stage flow */}
      <div className="px-5 py-4">
        <div className="relative">
          {stages.map((s, i) => (
            <div key={s.title} className="flex gap-3 relative pb-5 last:pb-0">
              {i < stages.length - 1 && (
                <span className="absolute left-2 top-5 bottom-0 w-px bg-zinc-200 -translate-x-1/2" />
              )}
              <StatusDot status={s.status} />
              <div className="flex-1 -mt-0.5 min-w-0">
                <div className="flex items-center gap-1.5">
                  <span className={`text-xs font-medium ${s.status === "idle" ? "text-zinc-400" : "text-accent"}`}>
                    {s.title}
                  </span>
                  {s.algo && <AlgoBadge id={s.algo} />}
                </div>
                <div className="text-[11px] text-muted mt-0.5 leading-relaxed">{s.detail}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Legend */}
      <div className="mt-auto px-5 py-4 border-t border-border">
        <p className="text-[10px] font-medium text-muted uppercase tracking-wide mb-2">
          Free-tier workarounds
        </p>
        <div className="space-y-1.5 text-[11px] text-zinc-600">
          <p className="flex items-center gap-1.5"><AlgoBadge id="A1" /> PCA router — picks temperature + path</p>
          <p className="flex items-center gap-1.5"><AlgoBadge id="A2" /> Trie guard — stands in for logit_bias</p>
          <p className="flex items-center gap-1.5"><AlgoBadge id="A3" /> SVD penalty — for frequency_penalty</p>
          <p className="flex items-center gap-1.5"><AlgoBadge id="A4" /> Vote ensemble — for logprobs</p>
        </div>
      </div>
    </aside>
  );
}
