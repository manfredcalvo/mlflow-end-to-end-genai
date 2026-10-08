import React, { useEffect, useRef, useState } from "react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import {
  Send,
  Loader2,
  ThumbsUp,
  ThumbsDown,
  ExternalLink,
  Sparkles,
} from "lucide-react";
import { CustomerSelector } from "@/components/telco/CustomerSelector";
import {
  TelcoChatMessage,
  TelcoReasoningStep,
} from "@/components/telco/types";
import {
  parseTelcoSseChunk,
  summarizeArgs,
  summarizeOutput,
} from "@/lib/telcoStream";
import {
  postTelcoFeedback,
  useTelcoExperiment,
} from "@/queries/useQueryTelco";
import { Spinner } from "@/components/Spinner";

interface TelcoChatProps {
  onActiveMessageChange?: (msg: TelcoChatMessage | undefined) => void;
}

export function TelcoChat({ onActiveMessageChange }: TelcoChatProps) {
  const [customerId, setCustomerId] = useState<string>("CUST-00015");
  const [messages, setMessages] = useState<TelcoChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const { data: experiment } = useTelcoExperiment();

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  useEffect(() => {
    const last = lastAssistant(messages);
    onActiveMessageChange?.(last);
  }, [messages, onActiveMessageChange]);

  async function handleSend() {
    const message = input.trim();
    if (!message || sending) return;
    setError(null);

    const userMsg: TelcoChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: message,
    };
    const assistantId = crypto.randomUUID();
    const assistantMsg: TelcoChatMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      isStreaming: true,
      reasoning: [],
    };

    const history = messages
      .filter((m) => !m.isStreaming)
      .map((m) => ({ role: m.role, content: m.content }));

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setInput("");
    setSending(true);

    try {
      const res = await fetch("/api/telco/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message,
          customer_id: customerId,
          conversation_history: history,
        }),
      });
      if (!res.ok || !res.body) {
        throw new Error(`HTTP ${res.status}`);
      }
      await consumeStream(res.body, assistantId);
    } catch (err: any) {
      setError(err?.message ?? "Failed to send message");
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? {
                ...m,
                isStreaming: false,
                content:
                  m.content ||
                  "Sorry, something went wrong. Please try again.",
              }
            : m,
        ),
      );
    } finally {
      setSending(false);
    }
  }

  async function consumeStream(body: ReadableStream<Uint8Array>, assistantId: string) {
    const reader = body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value);
      buffer = parseTelcoSseChunk(chunk, buffer, (evt) => {
        applyEventToMessage(setMessages, assistantId, evt);
      });
    }
  }

  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between gap-3">
          <div>
            <CardTitle className="flex items-center gap-2 text-base">
              <Sparkles className="h-4 w-4 text-primary" />
              Telco Support Agent
            </CardTitle>
            <p className="text-xs text-muted-foreground mt-1">
              Powered by MLflow tracing — every response is observable in
              production.
            </p>
          </div>
          <CustomerSelector value={customerId} onChange={setCustomerId} />
        </div>
      </CardHeader>
      <CardContent className="flex-1 overflow-hidden p-0">
        <div className="h-full flex flex-col">
          <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3">
            {messages.length === 0 && <EmptyChatState />}
            {messages.map((m) => (
              <MessageBubble
                key={m.id}
                message={m}
                experimentUrl={experiment?.mlflow_url ?? null}
              />
            ))}
            <div ref={bottomRef} />
          </div>
          <div className="border-t p-3 space-y-2">
            {error && (
              <Alert variant="destructive">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}
            <div className="flex gap-2">
              <Textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask about a customer's account, billing, or device…"
                rows={2}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleSend();
                  }
                }}
                className="flex-1 resize-none"
                disabled={sending}
              />
              <Button onClick={handleSend} disabled={sending || !input.trim()}>
                {sending ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Send className="h-4 w-4" />
                )}
              </Button>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function lastAssistant(messages: TelcoChatMessage[]): TelcoChatMessage | undefined {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "assistant") return messages[i];
  }
  return undefined;
}

function applyEventToMessage(
  setMessages: React.Dispatch<React.SetStateAction<TelcoChatMessage[]>>,
  assistantId: string,
  evt: import("@/components/telco/types").TelcoStreamEvent,
) {
  setMessages((prev) =>
    prev.map((m) => {
      if (m.id !== assistantId) return m;
      const reasoning = [...(m.reasoning ?? [])];
      switch (evt.type) {
        case "routing": {
          const step: TelcoReasoningStep = {
            kind: "routing",
            title: `Routed to ${evt.agent_type ?? "specialist"} agent`,
            detail: evt.routing_decision,
          };
          return { ...m, agentType: evt.agent_type, reasoning: [...reasoning, step] };
        }
        case "tool_call": {
          const step: TelcoReasoningStep = {
            kind: "tool_call",
            title: `Calling tool: ${evt.tool_name}`,
            callId: evt.call_id,
            argsPreview: summarizeArgs(evt.arguments),
          };
          return { ...m, reasoning: [...reasoning, step] };
        }
        case "tool_result": {
          const step: TelcoReasoningStep = {
            kind: "tool_result",
            title: "Tool result",
            callId: evt.call_id,
            outputPreview: summarizeOutput(evt.output),
          };
          return { ...m, reasoning: [...reasoning, step] };
        }
        case "response_text":
          return { ...m, content: evt.text };
        case "completion":
          return {
            ...m,
            content: evt.final_response || m.content,
            isStreaming: false,
            traceId: evt.trace_id,
            agentType: evt.agent_type ?? m.agentType,
          };
        case "error":
          return {
            ...m,
            isStreaming: false,
            content: m.content || `Error: ${evt.error}`,
          };
        default:
          return m;
      }
    }),
  );
}

function EmptyChatState() {
  return (
    <div className="py-16 text-center space-y-3 text-sm text-muted-foreground">
      <Sparkles className="h-8 w-8 mx-auto text-primary/50" />
      <p className="font-medium text-foreground">
        Ask the support agent anything
      </p>
      <p className="text-xs max-w-md mx-auto">
        Try: "What's my current bill?", "Why is my data slow?", or "Compare
        plans for me." Watch the right panel to see how the agent reasons,
        captured live by MLflow.
      </p>
    </div>
  );
}

function MessageBubble({
  message,
  experimentUrl,
}: {
  message: TelcoChatMessage;
  experimentUrl: string | null;
}) {
  const isUser = message.role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[85%] rounded-lg px-3 py-2 text-sm whitespace-pre-wrap ${
          isUser
            ? "bg-primary text-primary-foreground"
            : "bg-muted text-foreground"
        }`}
      >
        {message.content || (message.isStreaming ? <Spinner /> : null)}
        {!isUser && !message.isStreaming && message.traceId && (
          <AssistantFooter
            traceId={message.traceId}
            experimentUrl={experimentUrl}
          />
        )}
      </div>
    </div>
  );
}

function AssistantFooter({
  traceId,
  experimentUrl,
}: {
  traceId: string;
  experimentUrl: string | null;
}) {
  const [submitted, setSubmitted] = useState<"up" | "down" | null>(null);
  // Vote picked but not sent yet — a free-text comment box appears for it.
  const [pending, setPending] = useState<"up" | "down" | null>(null);
  const [comment, setComment] = useState("");
  const [sending, setSending] = useState(false);
  const traceUrl = experimentUrl
    ? `${experimentUrl}/traces?selectedEvaluationId=${traceId}`
    : null;

  async function send(isPositive: boolean, text: string) {
    setSending(true);
    try {
      await postTelcoFeedback({
        trace_id: traceId,
        is_positive: isPositive,
        comment: text.trim() || undefined,
      });
      setSubmitted(isPositive ? "up" : "down");
    } catch (e) {
      // eslint-disable-next-line no-console
      console.warn("feedback failed", e);
    } finally {
      setPending(null);
      setComment("");
      setSending(false);
    }
  }

  return (
    <div className="mt-2 pt-2 border-t border-border/40 flex flex-col gap-1 text-xs text-muted-foreground">
      <div className="flex items-center gap-1">
        <Button
          variant="ghost"
          size="sm"
          className="h-6 px-2"
          disabled={submitted !== null}
          onClick={() => setPending("up")}
        >
          <ThumbsUp
            className={`h-3 w-3 ${
              submitted === "up" || pending === "up"
                ? "text-emerald-600"
                : ""
            }`}
          />
        </Button>
        <Button
          variant="ghost"
          size="sm"
          className="h-6 px-2"
          disabled={submitted !== null}
          onClick={() => setPending("down")}
        >
          <ThumbsDown
            className={`h-3 w-3 ${
              submitted === "down" || pending === "down"
                ? "text-rose-600"
                : ""
            }`}
          />
        </Button>
        {traceUrl && (
          <Button
            variant="ghost"
            size="sm"
            className="h-6 px-2 ml-auto"
            onClick={() => window.open(traceUrl, "_blank")}
          >
            <ExternalLink className="h-3 w-3 mr-1" />
            Trace
          </Button>
        )}
        <Badge variant="outline" className="ml-2 text-[10px] font-mono">
          {traceId.slice(0, 12)}…
        </Badge>
      </div>
      {pending && (
        <div className="flex items-center gap-1">
          <Input
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            placeholder={
              pending === "up"
                ? "What went well? (optional)"
                : "What went wrong? (optional)"
            }
            className="h-6 flex-1 text-xs"
            autoFocus
            disabled={sending}
            onKeyDown={(e) => {
              if (e.key === "Enter") send(pending === "up", comment);
            }}
          />
          <Button
            variant="ghost"
            size="sm"
            className="h-6 px-2"
            disabled={sending}
            onClick={() => send(pending === "up", comment)}
          >
            <Send className="h-3 w-3" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="h-6 px-2"
            disabled={sending}
            onClick={() => send(pending === "up", "")}
          >
            Skip
          </Button>
        </div>
      )}
    </div>
  );
}
