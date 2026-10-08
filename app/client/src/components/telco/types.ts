export interface TelcoCustomer {
  customer_id: string;
  display_name: string;
}

export interface TelcoExperimentInfo {
  experiment_id: string;
  experiment_path: string;
  mlflow_url: string | null;
  environment: string;
}

export interface TelcoToolCall {
  type: "tool_call";
  tool_name: string;
  call_id: string;
  arguments: string;
  output?: string;
}

export interface TelcoToolResult {
  type: "tool_result";
  call_id: string;
  output: string;
}

export interface TelcoRoutingEvent {
  type: "routing";
  agent_type: string | null;
  routing_decision: string;
}

export interface TelcoResponseTextEvent {
  type: "response_text";
  text: string;
}

export interface TelcoCompletionEvent {
  type: "completion";
  agent_type: string | null;
  routing_decision: string | null;
  tools_used: TelcoToolCall[];
  final_response: string;
  trace_id: string | null;
  done: true;
}

export interface TelcoErrorEvent {
  type: "error";
  error: string;
  done?: boolean;
}

export type TelcoStreamEvent =
  | TelcoRoutingEvent
  | TelcoToolCall
  | TelcoToolResult
  | TelcoResponseTextEvent
  | TelcoCompletionEvent
  | TelcoErrorEvent;

export interface TelcoReasoningStep {
  kind: "routing" | "tool_call" | "tool_result";
  title: string;
  detail?: string;
  callId?: string;
  argsPreview?: string;
  outputPreview?: string;
}

export interface TelcoChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  isStreaming?: boolean;
  traceId?: string | null;
  agentType?: string | null;
  reasoning?: TelcoReasoningStep[];
}
