"use client";
import ReactMarkdown from "react-markdown";
import { Message } from "@/lib/types";
import { RouteStrip } from "./RouteStrip";

interface Props {
  message: Message;
}

export function MessageBubble({ message }: Props) {
  const { role, content, routeInfo, streaming, failed } = message;

  if (role === "user") {
    return (
      <div className="mb-5 flex justify-end">
        <div className="max-w-[80%] rounded-2xl rounded-tr-sm bg-accent px-4 py-2.5 text-sm leading-relaxed text-white">
          {content}
        </div>
      </div>
    );
  }

  const emptyFinished = !streaming && content.trim() === "";

  return (
    <div className="mb-6 flex justify-start">
      <div className="w-full min-w-0">
        <div className="rounded-2xl rounded-tl-sm border border-border bg-panel px-4 py-3 shadow-sm">
          {emptyFinished ? (
            <p className="text-sm text-muted">
              {failed
                ? "No answer was produced. See diagnostics below for what the pipeline reported."
                : "No content returned."}
            </p>
          ) : (
            <div className="prose prose-sm max-w-none text-sm leading-relaxed text-zinc-800 prose-pre:bg-zinc-900 prose-pre:text-zinc-100">
              <ReactMarkdown>{content || " "}</ReactMarkdown>
              {streaming && (
                <span className="ml-0.5 inline-block h-4 w-0.5 animate-pulse bg-zinc-400 align-middle" />
              )}
            </div>
          )}
        </div>

        {routeInfo && !streaming && <RouteStrip info={routeInfo} />}
      </div>
    </div>
  );
}
