"use client";
import { RouteInfo } from "@/lib/types";

function Metric({ label, value, tone }: { label: string; value: React.ReactNode; tone?: string }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-[9px] font-medium uppercase tracking-wider text-zinc-400 leading-none">{label}</span>
      <span className={`font-mono text-[12px] leading-none tnum ${tone ?? "text-accent"}`}>{value}</span>
    </div>
  );
}

function Dot({ color }: { color: string }) {
  return <span className="inline-block h-1.5 w-1.5 rounded-full align-middle" style={{ backgroundColor: color }} />;
}

const ALGOS = [
  { id: "A1", name: "router" },
  { id: "A2", name: "trie" },
  { id: "A3", name: "svd" },
  { id: "A4", name: "vote" },
];

export function RouteStrip({ info }: { info: RouteInfo }) {
  const pipeline = info.regime !== undefined;

  // Fallback / no-pipeline answers carry no routing telemetry; say so plainly.
  if (!pipeline) {
    return (
      <div className="mt-2 inline-flex items-center gap-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-1.5">
        <Dot color="#f59e0b" />
        <span className="font-mono text-[11px] text-amber-700">fallback · pipeline bypassed · {info.latencyMs > 0 ? `${info.latencyMs}ms` : "direct model call"}</span>
      </div>
    );
  }

  const agentic = info.path === "agentic";
  const regimeColor = info.regime === "logic" ? "#0ea5e9" : info.regime === "creative" ? "#8b5cf6" : "#71717a";
  const engaged = new Set(["A1", "A2", "A3", ...(agentic ? ["A4"] : [])]);

  return (
    <div className="mt-2 rounded-lg border border-border bg-panel px-3 py-2 shadow-sm">
      <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
        <Metric
          label="path"
          tone={agentic ? "text-amber-600" : "text-accent"}
          value={
            <span className="inline-flex items-center gap-1.5">
              <Dot color={agentic ? "#f59e0b" : "#10b981"} />
              {info.path}
            </span>
          }
        />
        <Metric
          label="regime"
          value={
            <span className="inline-flex items-center gap-1.5">
              <Dot color={regimeColor} />
              {info.regime}
            </span>
          }
        />
        {typeof info.confidence === "number" && <Metric label="confidence" value={`${Math.round(info.confidence * 100)}%`} />}
        <Metric label="temp" value={info.temperature.toFixed(2)} />
        {agentic && <Metric label="hops" value={info.hops} />}
        {agentic && typeof info.chunks === "number" && <Metric label="chunks" value={info.chunks} />}
        {agentic && typeof info.freqPenalty === "number" && <Metric label="redundancy" value={info.freqPenalty.toFixed(2)} />}
        {info.hallucinationInterceptions > 0 && (
          <Metric label="corrections" tone="text-amber-600" value={info.hallucinationInterceptions} />
        )}
        <Metric label="latency" value={`${info.latencyMs}ms`} />

        <div className="ml-auto flex items-center gap-1">
          {ALGOS.map((a) => {
            const on = engaged.has(a.id);
            return (
              <span
                key={a.id}
                title={`${a.id} ${a.name}${on ? " — engaged" : " — not used"}`}
                className={`rounded px-1 py-px font-mono text-[9px] font-semibold leading-none ${
                  on ? "bg-accent text-white" : "bg-faint text-zinc-300"
                }`}
              >
                {a.id}
              </span>
            );
          })}
        </div>
      </div>
    </div>
  );
}
