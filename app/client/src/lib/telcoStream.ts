import { TelcoStreamEvent } from "@/components/telco/types";

export interface SseLineHandler {
  (event: TelcoStreamEvent): void;
}

export function parseTelcoSseChunk(
  chunk: string,
  buffer: string,
  onEvent: SseLineHandler,
): string {
  const combined = buffer + chunk;
  const lines = combined.split("\n");
  // Last entry may be a partial line; preserve as new buffer.
  const carry = lines.pop() ?? "";

  for (const raw of lines) {
    const line = raw.trim();
    if (!line || line.startsWith(":")) continue;
    if (!line.startsWith("data: ")) continue;
    const data = line.slice(6);
    if (data === "[DONE]") continue;
    try {
      const evt = JSON.parse(data) as TelcoStreamEvent;
      onEvent(evt);
    } catch (e) {
      // Skip malformed events; the backend has its own recovery path.
      // eslint-disable-next-line no-console
      console.warn("Failed to parse telco SSE line:", e);
    }
  }
  return carry;
}

export function summarizeArgs(argsJson: string): string {
  try {
    const obj = JSON.parse(argsJson);
    const entries = Object.entries(obj).slice(0, 4);
    if (!entries.length) return "—";
    return entries
      .map(([k, v]) => `${k}=${truncate(JSON.stringify(v), 40)}`)
      .join(", ");
  } catch {
    return truncate(argsJson, 80);
  }
}

export function summarizeOutput(out: string | undefined): string {
  if (!out) return "";
  return truncate(out.replace(/\s+/g, " "), 160);
}

function truncate(s: string, n: number): string {
  if (s.length <= n) return s;
  return s.slice(0, n) + "…";
}
