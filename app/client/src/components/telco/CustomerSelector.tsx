import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useTelcoCustomers } from "@/queries/useQueryTelco";
import { Label } from "@/components/ui/label";
import { Loader2 } from "lucide-react";

interface CustomerSelectorProps {
  value: string;
  onChange: (customerId: string) => void;
}

export function CustomerSelector({ value, onChange }: CustomerSelectorProps) {
  const { data, isLoading, error } = useTelcoCustomers();

  return (
    <div className="flex items-center gap-2">
      <Label className="text-xs text-muted-foreground">Customer</Label>
      {isLoading ? (
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <Loader2 className="h-3 w-3 animate-spin" />
          Loading…
        </div>
      ) : error ? (
        <span className="text-xs text-destructive">Failed to load</span>
      ) : (
        <Select value={value} onValueChange={onChange}>
          <SelectTrigger className="h-8 w-[180px] text-xs">
            <SelectValue placeholder="Pick a customer" />
          </SelectTrigger>
          <SelectContent>
            {(data ?? []).map((c) => (
              <SelectItem key={c.customer_id} value={c.customer_id}>
                {c.display_name} ({c.customer_id})
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      )}
    </div>
  );
}
