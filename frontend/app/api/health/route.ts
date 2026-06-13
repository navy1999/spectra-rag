import { NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL ?? "http://localhost:8080";

// Lightweight status read for the system badge. Reports whether the spectra Go
// pipeline is reachable, and if not, whether a direct-provider fallback is
// configured. Never throws to the client.
export async function GET() {
  try {
    const r = await fetch(`${BACKEND_URL}/health`, {
      signal: AbortSignal.timeout(4000),
      cache: "no-store",
    });
    if (!r.ok) throw new Error(`health ${r.status}`);
    const j = await r.json();
    return NextResponse.json({
      status: "pipeline",
      model: j.model ?? null,
      mock: !!j.mock,
    });
  } catch {
    return NextResponse.json({
      status: process.env.OPENROUTER_API_KEY ? "fallback" : "unavailable",
      model: process.env.OPENROUTER_API_KEY ? process.env.DEFAULT_MODEL ?? "openai/gpt-oss-120b:free" : null,
      mock: false,
    });
  }
}
