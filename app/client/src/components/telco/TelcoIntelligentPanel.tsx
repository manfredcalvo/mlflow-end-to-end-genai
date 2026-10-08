import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Brain,
  Wrench,
  CheckCircle2,
  Route as RouteIcon,
  ExternalLink,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { TelcoChatMessage, TelcoReasoningStep } from "@/components/telco/types";
import { useTelcoExperiment } from "@/queries/useQueryTelco";

interface TelcoIntelligentPanelProps {
  activeMessage?: TelcoChatMessage;
}

export function TelcoIntelligentPanel({
  activeMessage,
}: TelcoIntelligentPanelProps) {
  const { data: experiment } = useTelcoExperiment();
  const traceUrl =
    activeMessage?.traceId && experiment?.mlflow_url
      ? `${experiment.mlflow_url}/traces?selectedEvaluationId=${activeMessage.traceId}`
      : null;

  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between gap-2">
          <CardTitle className="flex items-center gap-2 text-base">
            <Brain className="h-4 w-4 text-primary" />
            Agent Reasoning
          </CardTitle>
          {activeMessage?.agentType && (
            <Badge variant="secondary" className="text-xs">
              {activeMessage.agentType}
            </Badge>
          )}
        </div>
        <p className="text-xs text-muted-foreground">
          Each step below is captured live by MLflow tracing.
        </p>
      </CardHeader>
      <CardContent className="flex-1 overflow-hidden p-0">
        <ScrollArea className="h-full px-4 pb-4">
          {!activeMessage ? (
            <EmptyState />
          ) : !activeMessage.reasoning?.length ? (
            <div className="text-xs text-muted-foreground py-6">
              Waiting for the agent to begin…
            </div>
          ) : (
            <div className="space-y-3">
              {activeMessage.reasoning.map((step, idx) => (
                <ReasoningCard key={idx} step={step} />
              ))}
            </div>
          )}

          {traceUrl && (
            <div className="pt-4 mt-4 border-t">
              <Button
                variant="ghost"
                size="sm"
                className="w-full justify-start"
                onClick={() => window.open(traceUrl, "_blank")}
              >
                <ExternalLink className="h-3 w-3 mr-2" />
                View full trace in MLflow
              </Button>
            </div>
          )}
        </ScrollArea>
      </CardContent>
    </Card>
  );
}

function EmptyState() {
  return (
    <div className="py-12 text-center text-sm text-muted-foreground space-y-2">
      <Brain className="h-8 w-8 mx-auto text-muted-foreground/40" />
      <p>Send a message to see how the agent reasons about it.</p>
      <p className="text-xs">
        Routing decisions, tool calls, and tool results show up here as they
        happen — same data MLflow captures in production.
      </p>
    </div>
  );
}

function ReasoningCard({ step }: { step: TelcoReasoningStep }) {
  const { icon, color } = iconForKind(step.kind);
  return (
    <div className="rounded-md border bg-card p-3">
      <div className="flex items-start gap-2">
        <div className={`mt-0.5 ${color}`}>{icon}</div>
        <div className="flex-1 min-w-0">
          <div className="text-xs font-medium">{step.title}</div>
          {step.detail && (
            <div className="text-xs text-muted-foreground mt-1">
              {step.detail}
            </div>
          )}
          {step.argsPreview && (
            <pre className="text-[11px] mt-2 p-2 bg-muted rounded overflow-x-auto">
              {step.argsPreview}
            </pre>
          )}
          {step.outputPreview && (
            <pre className="text-[11px] mt-2 p-2 bg-muted rounded overflow-x-auto whitespace-pre-wrap">
              {step.outputPreview}
            </pre>
          )}
        </div>
      </div>
    </div>
  );
}

function iconForKind(kind: TelcoReasoningStep["kind"]) {
  switch (kind) {
    case "routing":
      return {
        icon: <RouteIcon className="h-4 w-4" />,
        color: "text-blue-600",
      };
    case "tool_call":
      return { icon: <Wrench className="h-4 w-4" />, color: "text-amber-600" };
    case "tool_result":
      return {
        icon: <CheckCircle2 className="h-4 w-4" />,
        color: "text-emerald-600",
      };
    default:
      return { icon: <Brain className="h-4 w-4" />, color: "text-primary" };
  }
}
