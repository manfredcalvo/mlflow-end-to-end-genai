import { useQuery } from "@tanstack/react-query";
import {
  TelcoCustomer,
  TelcoExperimentInfo,
} from "@/components/telco/types";

export function useTelcoCustomers() {
  return useQuery<TelcoCustomer[]>({
    queryKey: ["telco-customers"],
    queryFn: async () => {
      const res = await fetch("/api/telco/customers");
      if (!res.ok) throw new Error(`Failed to load customers (${res.status})`);
      return res.json();
    },
    staleTime: 60_000,
  });
}

export function useTelcoExperiment() {
  return useQuery<TelcoExperimentInfo>({
    queryKey: ["telco-experiment"],
    queryFn: async () => {
      const res = await fetch("/api/telco/mlflow-experiment");
      if (!res.ok)
        throw new Error(`Failed to load experiment info (${res.status})`);
      return res.json();
    },
    staleTime: 60_000,
  });
}

export interface TelcoFeedbackBody {
  trace_id: string;
  is_positive: boolean;
  comment?: string;
  /** Optional fallback reviewer — the backend prefers the logged-in user
   *  (X-Forwarded-Email header injected by the Databricks Apps proxy). */
  agent_id?: string;
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
