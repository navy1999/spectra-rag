"use client";
import ReactMarkdown from "react-markdown";
import { Message } from "@/lib/types";

interface Props { message: Message; }

export function MessageBubble({ message }: Props) {
  const { role, content, routeInfo, streaming } = message;

  if (role === "user") {
    return (
      <div className="flex justify-end mb-4">
        <div className="max-w-[70%] bg-accent text-white rounded-2xl rounded-tr-sm px-4 py-2.5 text-sm leading-relaxed">
          {content}
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start mb-4">
      <div className="max-w-[80%]">
        <div className="bg-white border border-border rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm">
          <div className="text-sm leading-relaxed text-zinc-800 prose prose-sm max-w-none">
            <ReactMarkdown>{content || " "}</ReactMarkdown>
            {streaming && (
              <span className="inline-block w-0.5 h-4 bg-zinc-400 ml-0.5 animate-pulse align-middle" />
            )}
          </div>
        </div>
        {routeInfo && !streaming && (
          <div className="mt-1.5 flex items-center gap-1.5 flex-wrap">
            {routeInfo.path === "chat" ? (
              <span className="text-xs text-zinc-400 bg-zinc-50 border border-zinc-200 rounded-full px-2.5 py-0.5">
                ⚡ Fast path · {routeInfo.latencyMs}ms
              </span>
            ) : (
              <span className="text-xs text-zinc-400 bg-zinc-50 border border-zinc-200 rounded-full px-2.5 py-0.5">
                🔍 Agentic · {routeInfo.hops} hop{routeInfo.hops !== 1 ? "s" : ""}
                {typeof routeInfo.chunks === "number" ? ` · ${routeInfo.chunks} chunks` : ""} · {routeInfo.latencyMs}ms
              </span>
            )}
            {routeInfo.regime && (
              <span
                className={`text-xs border rounded-full px-2.5 py-0.5 ${
                  routeInfo.regime === "logic"
                    ? "text-sky-700 bg-sky-50 border-sky-200"
                    : "text-violet-700 bg-violet-50 border-violet-200"
                }`}
              >
                {routeInfo.regime === "logic" ? "🧠" : "🎨"} {routeInfo.regime}
                {typeof routeInfo.confidence === "number"
                  ? ` · ${Math.round(routeInfo.confidence * 100)}% familiar`
                  : ""}
              </span>
            )}
            {routeInfo.temperature > 0 && (
              <span className="text-xs text-zinc-400 bg-zinc-50 border border-zinc-200 rounded-full px-2.5 py-0.5">
                temp {routeInfo.temperature.toFixed(2)}
              </span>
            )}
            {routeInfo.hallucinationInterceptions > 0 && (
              <span className="text-xs text-amber-600 bg-amber-50 border border-amber-200 rounded-full px-2.5 py-0.5">
                ✦ {routeInfo.hallucinationInterceptions} entity fix{routeInfo.hallucinationInterceptions !== 1 ? "es" : ""}
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
