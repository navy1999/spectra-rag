import { NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL ?? "http://localhost:8080";

// Active-graph stats (node/edge counts, per-type breakdown, and whether the
// graph is a user-ingested custom corpus).
export async function GET() {
  try {
    const res = await fetch(`${BACKEND_URL}/graph`, {
      cache: "no-store",
      signal: AbortSignal.timeout(10000),
    });
    const data = await res.json().catch(() => ({}));
    return NextResponse.json(data, { status: res.status });
  } catch {
    return NextResponse.json({}, { status: 502 });
  }
}
