"use client";

import {
  SidebarGroup,
  SidebarGroupLabel,
  SidebarGroupContent,
} from "@/components/ui/sidebar";
import { Spinner } from "@/components/Spinner";

export function NavDocuments({
  experiment,
  experimentIsLoading,
}: {
  experiment: any;
  experimentIsLoading: boolean;
}) {
  return (
    <SidebarGroup className="group-data-[collapsible=icon]:hidden">
      {/* <SidebarGroupLabel>Configuration</SidebarGroupLabel> */}
      <SidebarGroupContent className="space-y-4 px-2">
        <div className="space-y-2">
          <label className="text-xs font-medium text-sidebar-foreground/70">
            MLFlow Experiment
          </label>
          <div className="text-xs">
            {experimentIsLoading && <Spinner size={4} />}
            {!experimentIsLoading &&
              (experiment?.["mlflow_url"] ? (
                <a
                  href={experiment["mlflow_url"]}
                  className="text-blue-500 hover:underline break-all"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  Open experiment {experiment?.["experiment_id"]} ↗
                </a>
              ) : (
                <span className="text-sidebar-foreground/50">Unavailable</span>
              ))}
          </div>
        </div>
      </SidebarGroupContent>
    </SidebarGroup>
  );
}
