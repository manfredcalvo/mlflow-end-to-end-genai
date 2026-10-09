import "./index.css"; // Tailwind styles
import "./App.css";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";

import { useTelcoExperiment } from "@/queries/useQueryTelco";
import { TooltipProvider } from "@/components/ui/tooltip";
import { SidebarProvider } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/app-sidebar";
import { TelcoAssistant } from "@/components/telco/TelcoAssistant";
import { AnalyticsTab } from "@/components/analytics/AnalyticsTab";

const queryClient = new QueryClient();

export function Chat() {
  const { data: experiment, isLoading: experimentIsLoading } =
    useTelcoExperiment();

  return (
    <TooltipProvider>
      <SidebarProvider>
        <AppSidebar
          experiment={experiment}
          experimentIsLoading={experimentIsLoading}
        />
        <main className="flex-1 flex items-center justify-center p-4">
          <Routes>
            <Route
              path="/"
              element={
                <div className="w-full h-full">
                  <TelcoAssistant />
                </div>
              }
            />
            <Route
              path="/telco"
              element={
                <div className="w-full h-full">
                  <TelcoAssistant />
                </div>
              }
            />
            <Route
              path="/analytics"
              element={
                <div className="w-full h-full">
                  <AnalyticsTab />
                </div>
              }
            />
          </Routes>
        </main>
      </SidebarProvider>
    </TooltipProvider>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route path="/*" element={<Chat />} />
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;
