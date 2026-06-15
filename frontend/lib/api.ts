import { RouteInfo, SpectraMode, Diagnostic } from "./types";

export interface StreamCallbacks {
  onChunk: (text: string) => void;
  onRouteInfo: (info: RouteInfo) => void;
  onStatus: (mode: SpectraMode) => void;
  onDiagnostic: (diag: Diagnostic) => void;
  onDone: () => void;
  onError: (err: Error) => void;
}

export async function sendMessage(
  query: string,
  model: string | null,
  callbacks: StreamCallbacks,
  forceRetrieve = false
): Promise<void> {
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, model: model ?? undefined, force_retrieve: forceRetrieve }),
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const mode = (res.headers.get("X-Spectra-Mode") as SpectraMode) ?? "pipeline";
    callbacks.onStatus(mode);

    const routePath = res.headers.get("X-Route-Path") ?? "chat";
    const hopCount = parseInt(res.headers.get("X-Hop-Count") ?? "0");

    // Route path + hop count arrive as headers (known before streaming starts).
    // The leading `route` SSE event fills in full routing detail (regime,
    // confidence, PCA coordinates, centroids); the trailing `meta` event adds
    // latency + interception count, known only once the stream completes.
    const routeInfo: RouteInfo = {
      path: routePath as "chat" | "agentic",
      hops: hopCount,
      latencyMs: parseInt(res.headers.get("X-Latency-Ms") ?? "0"),
      temperature: 0,
      hallucinationInterceptions: 0,
    };
    if (mode === "pipeline") callbacks.onRouteInfo(routeInfo);

    const reader = res.body?.getReader();
    if (!reader) throw new Error("No response body");

    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const data = line.slice(6).trim();
        if (data === "[DONE]") {
          callbacks.onDone();
          return;
        }
        try {
          const parsed = JSON.parse(data);
          if (parsed.token) {
            callbacks.onChunk(parsed.token);
          } else if (parsed.route) {
            const r = parsed.route;
            routeInfo.path = r.path === "agentic" ? "agentic" : "chat";
            routeInfo.regime = r.regime;
            routeInfo.confidence = r.confidence;
            routeInfo.temperature = r.temperature ?? 0;
            routeInfo.pcaX = r.x;
            routeInfo.pcaY = r.y;
            routeInfo.distance = r.distance;
            routeInfo.hops = r.hops ?? routeInfo.hops;
            routeInfo.chunks = r.chunks;
            routeInfo.freqPenalty = r.freqPenalty;
            routeInfo.centroids = r.centroids;
            callbacks.onRouteInfo({ ...routeInfo });
          } else if (parsed.meta) {
            if (typeof parsed.meta.latencyMs === "number") routeInfo.latencyMs = parsed.meta.latencyMs;
            if (typeof parsed.meta.interceptions === "number")
              routeInfo.hallucinationInterceptions = parsed.meta.interceptions;
            callbacks.onRouteInfo({ ...routeInfo });
          } else if (parsed.error) {
            callbacks.onDiagnostic({
              level: "error",
              stage: parsed.error.stage,
              code: parsed.error.code,
              message: parsed.error.message ?? "Request failed.",
            });
          } else if (parsed.system) {
            if (parsed.system.mode) callbacks.onStatus(parsed.system.mode as SpectraMode);
            if (parsed.system.message)
              callbacks.onDiagnostic({ level: "warn", stage: "system", message: parsed.system.message });
          }
        } catch {
          // A line that isn't valid JSON is a keepalive or partial; ignore it
          // instead of dumping raw text into the answer.
        }
      }
    }
    callbacks.onDone();
  } catch (err) {
    callbacks.onDiagnostic({
      level: "error",
      stage: "transport",
      message: err instanceof Error ? err.message : String(err),
    });
    callbacks.onError(err instanceof Error ? err : new Error(String(err)));
  }
}
