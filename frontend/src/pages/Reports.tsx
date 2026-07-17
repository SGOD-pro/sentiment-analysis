/**
 * Reports page — matches Stitch "SentiMetric | Reports (v2)" screen.
 * Weekly performance stats, sentiment distribution bar chart, issue trend table.
 * All API calls scoped to batchId from useSessionStore.
 */
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Separator } from "@/components/ui/separator";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { CalendarDays, FileDown, Upload } from "lucide-react";
import { getCategoriesSummary, getIssuesDistribution } from "@/api/client";
import { useSessionStore } from "@/hooks/useSessionStore";
import type { CategorySummary, IssueCount } from "@/types";
import { DashboardPage } from "@/components/dashboard-layout";

const BAR_COLORS: Record<string, string> = {
  positive: "#05B169", neutral: "#F4B000", negative: "#CF202F",
};

export default function Reports() {
  const batchId = useSessionStore((s) => s.batchId);
  const navigate = useNavigate();
  const [categories, setCategories] = useState<CategorySummary[]>([]);
  const [issues, setIssues] = useState<IssueCount[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!batchId) { setLoading(false); return; }
    Promise.all([getCategoriesSummary(batchId), getIssuesDistribution(batchId)])
      .then(([c, i]) => {
        if (c.data) setCategories(c.data.categories);
        if (i.data) setIssues(i.data.issues);
      }).finally(() => setLoading(false));
  }, [batchId]);

  if (!batchId) {
    return (
      <DashboardPage sidebar={<div className="p-5"><p className="text-xs text-muted-foreground">No active session</p></div>}>
        <div className="flex-1 flex flex-col items-center justify-center gap-4 p-10 text-center">
          <Upload className="w-16 h-16 text-muted-foreground/30" />
          <h2 className="text-xl font-semibold">No data loaded</h2>
          <p className="text-sm text-muted-foreground">Upload a CSV to generate reports.</p>
          <Button onClick={() => navigate("/upload")} className="gap-2"><Upload className="w-4 h-4" />Import Data</Button>
        </div>
      </DashboardPage>
    );
  }

  // Derive distribution from categories
  const totPos = categories.reduce((s, c) => s + c.positive, 0);
  const totNeu = categories.reduce((s, c) => s + c.neutral, 0);
  const totNeg = categories.reduce((s, c) => s + c.negative, 0);
  const grand = totPos + totNeu + totNeg || 1;
  const distData = [
    { name: "POSITIVE", count: totPos, pct: Math.round(totPos / grand * 100) },
    { name: "NEUTRAL",  count: totNeu, pct: Math.round(totNeu / grand * 100) },
    { name: "NEGATIVE", count: totNeg, pct: Math.round(totNeg / grand * 100) },
  ];

  const totalProcessed = grand;
  const negTotal = issues.reduce((s, i) => s + i.count, 0) || 1;

  const reportsSidebar = (
    <div className="flex flex-col p-5 gap-4 h-full">
      <p className="font-semibold text-sm">Filters</p>
      <p className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground">Analysis Parameters</p>
      <Separator />
      <div>
        <p className="text-xs text-muted-foreground flex items-center gap-1.5"><CalendarDays className="w-3.5 h-3.5" /> Date Range</p>
        <Select defaultValue="7d">
          <SelectTrigger className="h-8 text-xs mt-2"><SelectValue /></SelectTrigger>
          <SelectContent>
            <SelectItem value="7d">Last 7 Days</SelectItem>
            <SelectItem value="30d">Last 30 Days</SelectItem>
            <SelectItem value="90d">Last 90 Days</SelectItem>
          </SelectContent>
        </Select>
      </div>
      <Button variant="outline" className="w-full mt-auto">Apply Filters</Button>
    </div>
  );

  return (
    <DashboardPage sidebar={reportsSidebar}>
      {/* Page header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Performance Reports</h1>
          <p className="text-xs text-muted-foreground mt-0.5">Last Update: {new Date().toLocaleDateString()}</p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm"><FileDown className="w-3.5 h-3.5 mr-1.5" />Export CSV</Button>
          <Button size="sm"><FileDown className="w-3.5 h-3.5 mr-1.5" />Export PDF</Button>
        </div>
      </div>

      {/* Weekly performance + model calibration */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        <Card className="lg:col-span-2">
          <CardHeader className="pb-2">
            <CardTitle className="text-base flex items-center gap-2">📊 Weekly Performance</CardTitle>
          </CardHeader>
          <CardContent>
            {loading ? <Skeleton className="h-24" /> : (
              <div className="grid grid-cols-3 gap-4">
                {[
                  { label: "TOTAL PROCESSED", value: totalProcessed.toLocaleString() },
                  { label: "AVG PROCESSING TIME", value: "—" },
                  { label: "SYSTEM UPTIME", value: "—" },
                ].map(({ label, value }) => (
                  <div key={label} className="bg-muted/30 rounded-lg p-4 border border-border">
                    <p className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">{label}</p>
                    <p className="text-3xl font-bold font-number mt-1">{value}</p>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2"><CardTitle className="text-base">Model Calibration</CardTitle></CardHeader>
          <CardContent className="space-y-3">
            {[
              { label: "Macro F1 Score", value: "0.79" },
              { label: "Negative Recall", value: "0.79" },
              { label: "Neutral Recall",  value: "0.74" },
            ].map(({ label, value }) => (
              <div key={label} className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">{label}</span>
                <span className="font-bold font-number">{value}</span>
              </div>
            ))}
            <Separator />
            <p className="text-xs text-muted-foreground">ℹ System maintains 95% confidence margin for sentiment extraction.</p>
          </CardContent>
        </Card>
      </div>

      {/* Sentiment distribution chart */}
      <Card>
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle className="text-base">Sentiment Distribution</CardTitle>
            <div className="flex gap-3">
              {[["positive", "Positive"], ["neutral", "Neutral"], ["negative", "Negative"]].map(([v, l]) => (
                <div key={v} className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-full" style={{ background: BAR_COLORS[v] }} />
                  <span className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">{l}</span>
                </div>
              ))}
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {loading ? <Skeleton className="h-64" /> : (
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={distData} margin={{ top: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="oklch(var(--border))" vertical={false} />
                <XAxis dataKey="name" tick={{ fill: "oklch(var(--foreground))", fontSize: 11 }} />
                <YAxis tick={{ fill: "oklch(var(--muted-foreground))", fontSize: 11 }} tickFormatter={(v) => `${v}%`} domain={[0, 100]} />
                <Tooltip 
                  cursor={{ fill: "oklch(var(--muted)/0.6)" }}
                  content={({ active, payload, label }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0];
                      const color = BAR_COLORS[data.payload.name.toLowerCase()];
                      return (
                        <div style={{ backgroundColor: "oklch(var(--card))", border: "1px solid oklch(var(--border))", borderRadius: 8, padding: "12px", color: "oklch(var(--foreground))" }}>
                          <p className="font-semibold text-sm mb-1">{label}</p>
                          <p className="text-sm" style={{ color }}>
                            Share : {data.value}%
                          </p>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Bar dataKey="pct" radius={[6, 6, 0, 0]} label={{ position: "top", fill: "oklch(var(--foreground))", fontSize: 12, fontWeight: 600 }}>
                  {distData.map((d) => <Cell key={d.name} fill={BAR_COLORS[d.name.toLowerCase()]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </CardContent>
      </Card>

      {/* Issue trend table */}
      <Card>
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle className="text-base">Issue Trend Analysis</CardTitle>
            <Badge variant="outline">Weekly Analysis</Badge>
          </div>
        </CardHeader>
        <CardContent className="px-0">
          {loading ? <Skeleton className="h-40 mx-6" /> : (
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border">
                  <th className="px-6 py-3 text-left text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Issue Tag</th>
                  <th className="px-6 py-3 text-right text-[10px] font-bold uppercase tracking-wider text-muted-foreground">This Period</th>
                  <th className="px-6 py-3 text-right text-[10px] font-bold uppercase tracking-wider text-muted-foreground">% of Negative</th>
                </tr>
              </thead>
              <tbody>
                {issues.map((issue) => (
                  <tr key={issue.issue_tag} className="border-b border-border hover:bg-accent/30 transition-colors">
                    <td className="px-6 py-3">
                      <Badge variant="secondary" className="font-normal text-xs">{issue.issue_tag.replaceAll("_", " ")}</Badge>
                    </td>
                    <td className="px-6 py-3 text-right font-number">{issue.count.toLocaleString()}</td>
                    <td className="px-6 py-3 text-right font-number text-muted-foreground">
                      {Math.round(issue.count / negTotal * 100)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </CardContent>
      </Card>
    </DashboardPage>
  );
}
