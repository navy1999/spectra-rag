export interface Centroid {
  name: string;
  x: number;
  y: number;
}

export interface ModelInfo {
  id: string;
  context: number;
  size: number;
}

// RouteInfo accumulates across the SSE stream: path/hops arrive as response
// headers, the leading `route` event fills in the full routing detail, and the
// trailing `meta` event adds latency + interception counts. Fields beyond the
// first three are absent when the Go backend is unreachable (direct-provider
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

// Topic-driven ingestion job status (GET /api/ingest/status).
export interface TopicStatus {
  state: "idle" | "running" | "done" | "error";
  topic?: string;
  stage?: string;
  papers?: number;
  nodes?: number;
  edges?: number;
  index_dim?: number;
  compressed?: boolean;
  error?: string;
}

// Active-graph stats (GET /api/graph).
export interface GraphStatus {
  nodes: number;
  edges: number;
  types?: Record<string, number>;
  custom?: boolean;
}

// Which operating mode produced a response. Drives the system status badge.
export type SpectraMode = "checking" | "pipeline" | "fallback" | "unavailable";

export interface SystemStatus {
  status: SpectraMode;
  model?: string | null;
  mock?: boolean;
}

// A non-answer signal (provider error, fallback notice) surfaced in the
// diagnostics UI rather than the answer body.
export interface Diagnostic {
  level: "error" | "warn" | "info";
  message: string;
  stage?: string;
  code?: number;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  routeInfo?: RouteInfo;
  streaming?: boolean;
  failed?: boolean;
}

export interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
}
