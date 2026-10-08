import { useQuery } from "react-query";
import {
  TelcoCustomer,
  TelcoExperimentInfo,
} from "@/components/telco/types";

export function useTelcoCustomers() {
  return useQuery<TelcoCustomer[]>(
    ["telco-customers"],
    async () => {
      const res = await fetch("/api/telco/customers");
      if (!res.ok) throw new Error(`Failed to load customers (${res.status})`);
      return res.json();
    },
    { staleTime: 60_000 },
  );
}

export function useTelcoExperiment() {
  return useQuery<TelcoExperimentInfo>(
    ["telco-experiment"],
    async () => {
      const res = await fetch("/api/telco/mlflow-experiment");
      if (!res.ok)
        throw new Error(`Failed to load experiment info (${res.status})`);
      return res.json();
    },
    { staleTime: 60_000 },
  );
}

export interface TelcoFeedbackBody {
  trace_id: string;
  is_positive: boolean;
  comment?: string;
  agent_id: string;
}

export async function postTelcoFeedback(body: TelcoFeedbackBody) {
  const res = await fetch("/api/telco/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`Feedback failed (${res.status})`);
  return res.json();
}
