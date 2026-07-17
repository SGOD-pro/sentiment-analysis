/**
 * Shared date-range picker using shadcn Calendar + Popover.
 * Returns { from, to } as ISO date strings or undefined.
 */
import { useState } from "react";
import { format, subDays } from "date-fns";
import { CalendarDays } from "lucide-react";
import { Calendar } from "@/components/ui/calendar";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import type { DateRange } from "react-day-picker";
import { cn } from "@/lib/utils";

export interface DateRangeValue {
  from?: string;  // ISO date
  to?: string;    // ISO date
}

const PRESETS = [
  { value: "all", label: "All Time" },
  { value: "7d", label: "Last 7 Days" },
  { value: "30d", label: "Last 30 Days" },
  { value: "90d", label: "Last 90 Days" },
  { value: "custom", label: "Custom Range" },
] as const;

function presetToRange(value: string): DateRange | undefined {
  const today = new Date();
  switch (value) {
    case "7d": return { from: subDays(today, 7), to: today };
    case "30d": return { from: subDays(today, 30), to: today };
    case "90d": return { from: subDays(today, 90), to: today };
    default: return undefined;
  }
}

function rangeToISO(range: DateRange | undefined): DateRangeValue {
  return {
    from: range?.from ? format(range.from, "yyyy-MM-dd") : undefined,
    to: range?.to ? format(range.to, "yyyy-MM-dd") : undefined,
  };
}

export function DateRangeFilter({ onChange }: { onChange: (v: DateRangeValue) => void }) {
  const [preset, setPreset] = useState("all");
  const [range, setRange] = useState<DateRange | undefined>();

  const handlePreset = (v: string) => {
    setPreset(v);
    if (v === "custom") return;
    const r = presetToRange(v);
    setRange(r);
    onChange(rangeToISO(r));
  };

  const handleCustom = (r: DateRange | undefined) => {
    setRange(r);
    onChange(rangeToISO(r));
  };

  const label = range?.from
    ? `${format(range.from, "MMM d")}${range.to ? ` – ${format(range.to, "MMM d")}` : ""}`
    : "Select dates";

  return (
    <div className="space-y-2">
      <p className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Date Range</p>
      <Select value={preset} onValueChange={handlePreset}>
        <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
        <SelectContent>
          {PRESETS.map((p) => <SelectItem key={p.value} value={p.value}>{p.label}</SelectItem>)}
        </SelectContent>
      </Select>
      {preset === "custom" && (
        <Popover>
          <PopoverTrigger asChild>
            <Button variant="outline" size="sm" className={cn("w-full justify-start text-left text-xs font-normal", !range?.from && "text-muted-foreground")}>
              <CalendarDays className="mr-2 h-3.5 w-3.5" />
              {label}
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-auto p-0" align="start">
            <Calendar mode="range" selected={range} onSelect={handleCustom} numberOfMonths={2} />
          </PopoverContent>
        </Popover>
      )}
    </div>
  );
}
