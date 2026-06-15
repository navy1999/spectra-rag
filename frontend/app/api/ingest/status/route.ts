import { NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL ?? "http://localhost:8080";

// Poll the current/last topic-ingestion job status from the Go backend.
export async function GET() {
  try {
    const res = await fetch(`${BACKEND_URL}/ingest/status`, {
      cache: "no-store",
      signal: AbortSignal.timeout(10000),
    });
    const data = await res.json().catch(() => ({}));
    return NextResponse.json(data, { status: res.status });
  } catch {
    return NextResponse.json({ state: "idle" }, { status: 502 });
  }
}
