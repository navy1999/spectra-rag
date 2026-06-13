import { NextResponse } from "next/server";

// Proxies OpenRouter's public model catalog (no key needed), keeps only free
// models, and returns them small-first so the picker leads with the models
// where the framework's effect is most visible. Cached for an hour.
export const revalidate = 3600;

interface ORModel {
  id: string;
  context_length?: number;
}

function sizeHint(id: string): number {
  const m = id.match(/(\d+(?:\.\d+)?)\s*b/i);
  return m ? parseFloat(m[1]) : Number.MAX_SAFE_INTEGER;
}

export async function GET() {
  try {
    const r = await fetch("https://openrouter.ai/api/v1/models", {
      signal: AbortSignal.timeout(8000),
      next: { revalidate: 3600 },
    });
    if (!r.ok) throw new Error(`models ${r.status}`);
    const data = (await r.json())?.data as ORModel[] | undefined;
    const free = (data ?? [])
      .filter((m) => m.id.endsWith(":free"))
      .map((m) => ({ id: m.id, context: m.context_length ?? 0, size: sizeHint(m.id) }))
      .sort((a, b) => a.size - b.size);
    return NextResponse.json({ models: free });
  } catch {
    // Fallback so the picker still works if the catalog is unreachable.
    return NextResponse.json({
      models: [
        { id: "openai/gpt-oss-20b:free", context: 131072, size: 20 },
        { id: "openai/gpt-oss-120b:free", context: 131072, size: 120 },
      ],
    });
  }
}
