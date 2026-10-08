import { useState } from "react";
import { TelcoChat } from "@/components/telco/TelcoChat";
import { TelcoIntelligentPanel } from "@/components/telco/TelcoIntelligentPanel";
import { TelcoChatMessage } from "@/components/telco/types";

export function TelcoAssistant() {
  const [activeMessage, setActiveMessage] = useState<TelcoChatMessage | undefined>(
    undefined,
  );

  return (
    <div className="w-full h-full flex flex-col gap-4 p-4">
      <div>
        <h1 className="text-xl font-semibold">BGP Bank — Customer Support</h1>
        <p className="text-sm text-muted-foreground max-w-2xl">
          A banking customer-support agent for BGP Bank, traced end-to-end by
          MLflow. Pick a customer and ask about their accounts, transactions,
          loans, credit cards, products, or branches.
        </p>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 flex-1 min-h-0">
        <div className="lg:col-span-8 min-h-0">
          <TelcoChat onActiveMessageChange={setActiveMessage} />
        </div>
        <div className="lg:col-span-4 min-h-0">
          <TelcoIntelligentPanel activeMessage={activeMessage} />
        </div>
      </div>
    </div>
  );
}
