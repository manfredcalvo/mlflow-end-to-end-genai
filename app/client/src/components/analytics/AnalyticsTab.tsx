import { useQueryDashboard } from "@/queries/useQueryDashboard";
import { Alert, AlertDescription } from "@/components/ui/alert";

/**
 * The Analytics tab: embeds the bundle-created AI/BI portfolio dashboard
 * (iframe to {host}/embed/dashboardsv3/<id>, rendered with the viewer's
 * existing workspace session — app users are already Databricks-authenticated).
 */
export function AnalyticsTab() {
  const { data: dashboard, isLoading, error } = useQueryDashboard();

  if (isLoading) {
    return (
      <div className="p-4 text-sm text-muted-foreground">
        Loading dashboard…
      </div>
    );
  }

  if (error || !dashboard?.embed_url) {
    return (
      <div className="p-4">
        <Alert>
          <AlertDescription>
            Dashboard not available — it is created by your Databricks bundle
            deploy. Run <code>databricks bundle deploy</code> and reload.
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  return (
    <div className="w-full h-full p-2">
      <iframe
        src={dashboard.embed_url}
        title="Banking Portfolio Dashboard"
        className="w-full h-full min-h-[85vh] rounded-lg border border-border"
        // The workspace embed surface is a full app; keep iframe chrome off.
        referrerPolicy="no-referrer-when-downgrade"
      />
    </div>
  );
}
