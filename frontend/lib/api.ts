import { RouteInfo } from "./types";

export interface StreamCallbacks {
  onChunk: (text: string) => void;
  onRouteInfo: (info: RouteInfo) => void;
  onDone: () => void;
  onError: (err: Error) => void;
}

export async function sendMessage(
  query: string,
  callbacks: StreamCallbacks
): Promise<void> {
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);

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
    callbacks.onRouteInfo(routeInfo);

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
        if (line.startsWith("data: ")) {
          const data = line.slice(6).trim();
          if (data === "[DONE]") { callbacks.onDone(); return; }
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
              if (typeof parsed.meta.interceptions === "number") routeInfo.hallucinationInterceptions = parsed.meta.interceptions;
              callbacks.onRouteInfo({ ...routeInfo });
            }
          } catch { callbacks.onChunk(data); }
        }
      }
    }
    callbacks.onDone();
  } catch (err) {
    callbacks.onError(err instanceof Error ? err : new Error(String(err)));
  }
}
