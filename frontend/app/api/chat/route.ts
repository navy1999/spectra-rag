import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL ?? "http://localhost:8080";

export async function POST(req: NextRequest) {
  const body = await req.json();
  const { query } = body;

  const encoder = new TextEncoder();
  const stream = new TransformStream();
  const writer = stream.writable.getWriter();

  const headers = new Headers({
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
  });

  // Try the Go backend first. Await the response headers (not the full body)
  // so X-Route-Path / X-Hop-Count / X-Latency-Ms can be forwarded to the
  // client — headers can't be added to a NextResponse after construction.
  let backendRes: Response | null = null;
  try {
    backendRes = await fetch(`${BACKEND_URL}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
      signal: AbortSignal.timeout(30000),
    });
    if (!backendRes.ok) throw new Error(`Backend ${backendRes.status}`);

    headers.set("X-Route-Path", backendRes.headers.get("X-Route-Path") ?? "chat");
    headers.set("X-Hop-Count", backendRes.headers.get("X-Hop-Count") ?? "0");
    headers.set("X-Route-Regime", backendRes.headers.get("X-Route-Regime") ?? "");
    headers.set("X-Route-Confidence", backendRes.headers.get("X-Route-Confidence") ?? "");
    headers.set("X-Freq-Penalty", backendRes.headers.get("X-Freq-Penalty") ?? "");
  } catch {
    backendRes = null;
  }

  (async () => {
    try {
      if (backendRes) {
        const reader = backendRes.body?.getReader();
        if (!reader) throw new Error("No body");

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          await writer.write(value);
        }
        return;
      }

      // Fallback: direct OpenRouter call
      const apiKey = process.env.OPENROUTER_API_KEY;
      if (!apiKey) {
        await writer.write(encoder.encode('data: {"token":"[Backend unavailable — set OPENROUTER_API_KEY or start the Go backend]"}\n\ndata: [DONE]\n\n'));
        return;
      }

      const orRes = await fetch("https://openrouter.ai/api/v1/chat/completions", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${apiKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: "openai/gpt-oss-120b:free",
          messages: [{ role: "user", content: query }],
          stream: true,
        }),
      });

      const reader = orRes.body?.getReader();
      if (reader) {
        const decoder = new TextDecoder();
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const text = decoder.decode(value, { stream: true });
          for (const line of text.split("\n")) {
            if (line.startsWith("data: ")) {
              const d = line.slice(6).trim();
              if (d === "[DONE]") { await writer.write(encoder.encode("data: [DONE]\n\n")); break; }
              try {
                const p = JSON.parse(d);
                const token = p.choices?.[0]?.delta?.content;
                if (token) await writer.write(encoder.encode(`data: ${JSON.stringify({ token })}\n\n`));
              } catch {}
            }
          }
        }
      }
    } catch {
      // best-effort fallback; nothing more to send if this also fails
    } finally {
      await writer.close();
    }
  })();

  return new NextResponse(stream.readable, { headers });
}
