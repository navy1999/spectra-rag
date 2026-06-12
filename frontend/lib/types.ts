export interface Centroid {
  name: string;
  x: number;
  y: number;
}

// RouteInfo accumulates across the SSE stream: path/hops arrive as response
// headers, the leading `route` event fills in the full routing detail, and the
// trailing `meta` event adds latency + interception counts. Fields beyond the
// first three are absent when the Go backend is unreachable (direct-OpenRouter
// fallback), so the UI must degrade gracefully.
export interface RouteInfo {
  path: "chat" | "agentic";
  hops: number;
  latencyMs: number;
  temperature: number;
  hallucinationInterceptions: number;
  regime?: string;
  confidence?: number;
  pcaX?: number;
  pcaY?: number;
  distance?: number;
  chunks?: number;
  freqPenalty?: number;
  centroids?: Centroid[];
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  routeInfo?: RouteInfo;
  streaming?: boolean;
}

export interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
}
