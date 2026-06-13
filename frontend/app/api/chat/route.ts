import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL ?? "http://localhost:8080";
const FALLBACK_MODEL = process.env.DEFAULT_MODEL ?? "openai/gpt-oss-120b:free";

function sse(obj: unknown): string {
  return `data: ${JSON.stringify(obj)}\n\n`;
}

export async function POST(req: NextRequest) {
  const body = await req.json();
  const { query, model } = body;

  const encoder = new TextEncoder();
  const stream = new TransformStream();
  const writer = stream.writable.getWriter();

  const headers = new Headers({
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
  });

  // Reach the Go backend first. Await the response head (not the body) so the
  // routing headers and the operating mode can be forwarded to the client;
  // headers can't be added to a NextResponse after construction.
  let backendRes: Response | null = null;
  try {
    backendRes = await fetch(`${BACKEND_URL}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, model }),
      signal: AbortSignal.timeout(60000),
    });
    if (!backendRes.ok) throw new Error(`Backend ${backendRes.status}`);

    headers.set("X-Spectra-Mode", "pipeline");
    headers.set("X-Route-Path", backendRes.headers.get("X-Route-Path") ?? "chat");
    headers.set("X-Hop-Count", backendRes.headers.get("X-Hop-Count") ?? "0");
    headers.set("X-Route-Regime", backendRes.headers.get("X-Route-Regime") ?? "");
    headers.set("X-Route-Confidence", backendRes.headers.get("X-Route-Confidence") ?? "");
    headers.set("X-Freq-Penalty", backendRes.headers.get("X-Freq-Penalty") ?? "");
  } catch {
    backendRes = null;
    headers.set("X-Spectra-Mode", process.env.OPENROUTER_API_KEY ? "fallback" : "unavailable");
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

      // The Go backend is unreachable. Fall back to a direct provider call if a
      // key is configured here; otherwise report it as a diagnostic, not an
      // answer.
      const apiKey = process.env.OPENROUTER_API_KEY;
      if (!apiKey) {
        await writer.write(
          encoder.encode(
            sse({
              error: {
                stage: "backend",
                code: 0,
                message:
                  "The spectra Go backend is unreachable and no fallback key is configured. Start the backend (cd backend && go run .) to run the full pipeline.",
              },
            }) + "data: [DONE]\n\n"
          )
        );
        return;
      }

      // Fallback mode: a plain provider call with none of the spectra pipeline.
      await writer.write(
        encoder.encode(
          sse({
            system: {
              mode: "fallback",
              message:
                "Go backend unreachable. Answering with a direct model call; routing, retrieval, and the control-surface algorithms are bypassed.",
            },
          })
        )
      );

      const orRes = await fetch("https://openrouter.ai/api/v1/chat/completions", {
        method: "POST",
        headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
        body: JSON.stringify({
          model: model || FALLBACK_MODEL,
          messages: [{ role: "user", content: query }],
          stream: true,
        }),
      });

      if (!orRes.ok) {
        await writer.write(
          encoder.encode(
            sse({ error: { stage: "openrouter", code: orRes.status, message: `Fallback provider returned HTTP ${orRes.status}.` } }) +
              "data: [DONE]\n\n"
          )
        );
        return;
      }

      const reader = orRes.body?.getReader();
      if (reader) {
        const decoder = new TextDecoder();
        let buf = "";
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });
          const lines = buf.split("\n");
          buf = lines.pop() ?? "";
          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const d = line.slice(6).trim();
            if (d === "[DONE]") {
              await writer.write(encoder.encode("data: [DONE]\n\n"));
              return;
            }
            try {
              const p = JSON.parse(d);
              const token = p.choices?.[0]?.delta?.content;
              if (token) await writer.write(encoder.encode(sse({ token })));
            } catch {
              /* ignore keepalives / partials */
            }
          }
        }
        await writer.write(encoder.encode("data: [DONE]\n\n"));
      }
    } catch (err) {
      await writer.write(
        encoder.encode(
          sse({ error: { stage: "proxy", code: 0, message: err instanceof Error ? err.message : "stream failed" } }) +
            "data: [DONE]\n\n"
        )
      );
    } finally {
      await writer.close();
    }
  })();

  return new NextResponse(stream.readable, { headers });
}
