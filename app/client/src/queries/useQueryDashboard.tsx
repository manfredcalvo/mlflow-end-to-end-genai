import { useQuery } from "@tanstack/react-query";

export interface DashboardInfo {
  dashboard_id: string;
  embed_url: string | null;
}

export function useQueryDashboard() {
  return useQuery<DashboardInfo>({
    queryKey: ["dashboard"],
    queryFn: async () => {
      const res = await fetch("/api/dashboard");
      if (!res.ok)
        throw new Error(`Failed to load dashboard info (${res.status})`);
      return res.json();
    },
    staleTime: 60_000,
  });
}
