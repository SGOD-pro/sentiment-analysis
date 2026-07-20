/**
 * Settings page — matches Stitch "SentiMetric | Settings (v2)" screen.
 * Confidence threshold slider, notification rules panel.
 */
import { useState } from "react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Input } from "@/components/ui/input";
import { Trash2, Plus, Power, ArrowLeft } from "lucide-react";

interface Rule { id: string; name: string; level: string; desc: string; threshold: number; enabled: boolean; }

const INITIAL_RULES: Rule[] = [
  { id: "1", name: "Sentiment Drops",    level: "CRITICAL", desc: "Alert when overall sentiment falls below a specific threshold.", threshold: 35, enabled: true },
  { id: "2", name: "Issue Spike Detection", level: "WARNING", desc: 'Notify team if "Negative Billing" tags increase rapidly.', threshold: 15, enabled: true },
  { id: "3", name: "Volume Surge",       level: "INFO",     desc: "Alert when daily review volume exceeds historical average.", threshold: 2.5, enabled: true },
  { id: "4", name: "Anomaly Detection",  level: "BETA",     desc: "AI-driven pattern matching for unexpected semantic shifts.", threshold: 0, enabled: false },
];

const LEVEL_VARIANT: Record<string, "default" | "destructive" | "secondary" | "outline"> = {
  CRITICAL: "destructive", WARNING: "default", INFO: "secondary", BETA: "outline",
};

export default function Settings() {
  const [confidence, setConfidence] = useState(0.85);
  const [excludeLow, setExcludeLow] = useState(true);
  const [autoTag, setAutoTag] = useState(false);
  const [rules, setRules] = useState<Rule[]>(INITIAL_RULES);

  const updateRule = (id: string, patch: Partial<Rule>) =>
    setRules((rs) => rs.map((r) => (r.id === id ? { ...r, ...patch } : r)));
  const removeRule = (id: string) => setRules((rs) => rs.filter((r) => r.id !== id));

  const save = () => toast.success("Configuration saved");
  const discard = () => { setConfidence(0.85); setExcludeLow(true); setAutoTag(false); setRules(INITIAL_RULES); };

  return (
    <div className="pt-14 min-h-screen">
      <main className="max-w-5xl mx-auto p-6 space-y-6">
        <div className="flex items-center gap-4">
          <Button variant="outline" size="icon" onClick={() => window.history.back()} className="shrink-0 fixed top-4 left-4 md:top-8 md:left-8">
            <ArrowLeft className="w-5 h-5" />
          </Button>
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">Global Settings</h1>
            <p className="text-sm text-muted-foreground mt-0.5">Configure system thresholds and automated notification rules for the sentiment engine.</p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
          {/* Confidence threshold */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base">⚙ Confidence Threshold</CardTitle>
            </CardHeader>
            <CardContent className="space-y-5">
              <div>
                <div className="flex justify-between items-center mb-3">
                  <span className="text-sm text-muted-foreground">Minimum Confidence Score</span>
                  <span className="font-bold font-number text-primary">{confidence.toFixed(2)}</span>
                </div>
                <Slider min={0} max={1} step={0.01} value={[confidence]} onValueChange={([v]) => setConfidence(v)} />
                <div className="flex justify-between text-[10px] text-muted-foreground mt-1">
                  <span>0.00</span><span>1.00</span>
                </div>
              </div>
              <Separator />
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium">Exclude low-confidence predictions</p>
                  <p className="text-xs text-muted-foreground">Discard any data below the set threshold</p>
                </div>
                <Switch checked={excludeLow} onCheckedChange={setExcludeLow} />
              </div>
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium">Auto-tagging</p>
                  <p className="text-xs text-muted-foreground">Apply sentiment tags automatically to high-confidence rows</p>
                </div>
                <Switch checked={autoTag} onCheckedChange={setAutoTag} />
              </div>
            </CardContent>
          </Card>

          {/* Notification rules */}
          <Card>
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <CardTitle className="text-base">🔔 Notification Rules</CardTitle>
                <Button variant="outline" size="sm"><Plus className="w-3.5 h-3.5 mr-1" />Create New Rule</Button>
              </div>
            </CardHeader>
            <CardContent className="space-y-3">
              {rules.map((rule) => (
                <div key={rule.id} className={`rounded-lg border p-4 space-y-2 transition-opacity ${rule.enabled ? "" : "opacity-50"}`} style={{ borderColor: "hsl(var(--border))" }}>
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-semibold">{rule.name}</span>
                        <Badge variant={LEVEL_VARIANT[rule.level] ?? "secondary"} className="text-[10px]">{rule.level}</Badge>
                      </div>
                      <p className="text-xs text-muted-foreground mt-0.5">{rule.desc}</p>
                    </div>
                    <div className="flex items-center gap-1.5 shrink-0">
                      {rule.enabled ? (
                        <><Input type="number" value={rule.threshold} onChange={(e) => updateRule(rule.id, { threshold: Number(e.target.value) })}
                            className="h-7 w-16 text-xs font-number text-center" />
                          <span className="text-xs text-muted-foreground">%</span>
                          <Button variant="ghost" size="sm" onClick={() => removeRule(rule.id)} className="h-7 w-7 p-0 text-muted-foreground">
                            <Trash2 className="w-3.5 h-3.5" />
                          </Button>
                        </>
                      ) : (
                        <Button variant="outline" size="sm" onClick={() => updateRule(rule.id, { enabled: true })} className="h-7 text-xs">
                          <Power className="w-3 h-3 mr-1" />Enable Rule
                        </Button>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        </div>

        <div className="flex gap-3 justify-end">
          <Button variant="outline" onClick={discard}>Discard Changes</Button>
          <Button onClick={save}>Save Configuration</Button>
        </div>
      </main>
    </div>
  );
}
