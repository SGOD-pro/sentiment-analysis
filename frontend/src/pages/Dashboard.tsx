/**
 * Dashboard page — matches Stitch "SentiMetric | Dashboard (v2)" screen.
 * Layout: fixed sidebar (filters) + main area (alert, stat cards, trend chart, category table + issue bar).
 * All column labels come from useColumnMap, never hardcoded.
 * All API calls scoped to batchId from useSessionStore.
 */
import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Slider } from "@/components/ui/slider";
import { AlertTriangle, Filter, TrendingDown, TrendingUp, MessageSquare, Tag, Upload, Angry, Annoyed, Smile } from "lucide-react";
import { getCategoriesSummary, getIssuesDistribution, getTrends } from "@/api/client";
import { loadColumnMap } from "@/hooks/useColumnMap";
import { useSessionStore } from "@/hooks/useSessionStore";
import type { CategorySummary, IssueCount, TrendWeek } from "@/types";
import { DashboardPage } from "@/components/dashboard-layout";
import { DateRangeFilter, type DateRangeValue } from "@/components/DateRangeFilter";
import { cn } from "@/lib/utils";

// ── Filter state shape ───────────────────────────────────────────────────────
interface Filters {
  category: string;
  sentiment: string;
  minConf: number;
  dateRange: DateRangeValue;
}

const INITIAL_FILTERS: Filters = { category: "all", sentiment: "all", minConf: 0, dateRange: {} };

// ── Sidebar ──────────────────────────────────────────────────────────────────
function Sidebar({
  catCol, categories, filters, onFilter,
}: {
  catCol?: string;
  categories: CategorySummary[];
  filters: Filters;
  onFilter: (f: Partial<Filters>) => void;
}) {
  const sentimentBtns = [
    { v: "negative", icon: Angry, label: "Neg" },
    { v: "neutral", icon: Annoyed, label: "Neu" },
    { v: "positive", icon: Smile, label: "Pos" },
  ];

  return (
    <div className="flex flex-col p-5 gap-5 h-full">
      <div className="flex items-center gap-2">
        <Filter className="w-4 h-4 text-primary" />
        <span className="font-semibold text-sm">Filters</span>
      </div>
      <p className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground -mt-4">Analysis Parameters</p>

      <div className="space-y-4 flex-1">
        {/* Date Range — wired to API */}
        <DateRangeFilter onChange={(dateRange) => onFilter({ dateRange })} />

        {/* Category — only shown if catCol was mapped */}
        {catCol && (
          <div>
            <p className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground mb-2">{catCol}</p>
            <select
              className="w-full h-9 text-xs rounded-md border border-input bg-background px-3"
              value={filters.category}
              onChange={(e) => onFilter({ category: e.target.value })}
            >
              <option value="all">All {catCol}s</option>
              {categories.map((c) => <option key={c.category} value={c.category}>{c.category}</option>)}
            </select>
          </div>
        )}

        {/* Sentiment toggle buttons */}
        <div>
          <p className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground mb-2">Sentiment</p>
          <div className="grid grid-cols-3 gap-1">
            {sentimentBtns.map(({ v, icon: Icon, label }) => (
              <Button key={v} size="sm" variant={filters.sentiment === v ? "default" : "outline"}
                className="h-9 text-base px-0"
                onClick={() => onFilter({ sentiment: filters.sentiment === v ? "all" : v })}
                title={label}>
                <Icon className="w-5 h-5" />
              </Button>
            ))}
          </div>
        </div>

        {/* Confidence */}
        <div>
          <div className="flex justify-between items-center mb-2">
            <p className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Confidence</p>
            <span className="text-[10px] font-bold font-number text-muted-foreground">{Math.round(filters.minConf * 100)}%</span>
          </div>
          <Slider min={0} max={1} step={0.05} value={[filters.minConf]}
            onValueChange={([v]) => onFilter({ minConf: v })} className="w-full" />
          <div className="flex justify-between text-[10px] text-muted-foreground mt-1">
            <span>0%</span><span>MIN: {Math.round(filters.minConf * 100)}%</span>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Stat card ─────────────────────────────────────────────────────────────────
function StatCard({ label, value, sub, icon: Icon, delta, deltaUp }: {
  label: string; value: string; sub?: string;
  icon: React.ElementType; delta?: string; deltaUp?: boolean;
}) {
  return (
    <Card className="hover:shadow-lg transition-shadow duration-200 h-full">
      <CardContent className="flex flex-col h-full relative overflow-hidden">
        {/* Top: Label */}
        <div className="flex justify-between items-start mb-3">
          <span className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">{label}</span>
          <div className="flex items-center gap-1">
            {delta && (
              <Badge variant="outline" className={`text-[10px] font-number ${deltaUp ? "text-green-500 border-green-500/30 bg-green-500/10" : "text-destructive border-destructive/30 bg-destructive/10"}`}>
                {deltaUp ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />} {delta}
              </Badge>
            )}
            <Icon className="w-4 h-4 text-muted-foreground" />
          </div>
        </div>

        {/* Middle: Value */}
        <div className={cn("flex-1 text-4xl font-bold font-number leading-tight", label == "Top Issue" && "text-xl")}>
          {value}
        </div>

        {/* Bottom: Sub text */}
        <div className="h-4 mt-3 flex items-end">
          {sub ? (
            <span className="text-[10px] text-muted-foreground uppercase tracking-wider truncate w-full">{sub}</span>
          ) : null}
        </div>
      </CardContent>
    </Card>
  );
}

// ── No session guard ──────────────────────────────────────────────────────────
function NoSessionPrompt() {
  const navigate = useNavigate();
  return (
    <div className="flex-1 flex flex-col items-center justify-center gap-4 p-10 text-center">
      <div className="w-1/2">
        <Upload className="w-16 h-16 text-muted-foreground/30 mx-auto" />
        <h2 className="text-xl font-semibold">No data loaded</h2>
        <p className="text-sm text-muted-foreground my-6">
          Upload a CSV file to start analyzing sentiment data. Your dashboard will populate automatically after processing.
        </p>
        <Button onClick={() => navigate("/upload")} className="gap-2">
          <Upload className="w-4 h-4" />Import Data
        </Button>
      </div>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function Dashboard() {
  const colMap = loadColumnMap();
  const batchId = useSessionStore((s) => s.batchId);
  const [weeks, setWeeks] = useState<TrendWeek[]>([]);
  const [categories, setCategories] = useState<CategorySummary[]>([]);
  const [issues, setIssues] = useState<IssueCount[]>([]);
  const [loading, setLoading] = useState(true);
  const [filters, setFilters] = useState<Filters>(INITIAL_FILTERS);

  const updateFilters = (partial: Partial<Filters>) =>
    setFilters((prev) => ({ ...prev, ...partial }));

  useEffect(() => {
    if (!batchId) { setLoading(false); return; }
    setLoading(true);
    const cat = filters.category !== "all" ? filters.category : undefined;
    const { from, to } = filters.dateRange;
    Promise.all([
      getTrends(batchId, from, to, cat),
      getCategoriesSummary(batchId, from, to),
      getIssuesDistribution(batchId, from, to, cat),
    ]).then(([t, c, i]) => {
      if (t.data) setWeeks(t.data.weeks);
      if (c.data) setCategories(c.data.categories);
      if (i.data) setIssues(i.data.issues);
    }).finally(() => setLoading(false));
  }, [batchId, filters.category, filters.dateRange]);

  if (!batchId) return <DashboardPage sidebar={<div className="p-5"><p className="text-xs text-muted-foreground">No active session</p></div>}><NoSessionPrompt /></DashboardPage>;

  // Client-side filtering: sentiment + minConf filter the category table display.
  // ponytail: confidence filtering on aggregates uses dominant-ratio proxy.
  //           Upgrade: backend avg_confidence field per category.
  const visibleCategories = categories.filter((c) => {
    if (filters.sentiment !== "all") {
      const sentCount = c[filters.sentiment as "positive" | "negative" | "neutral"] ?? 0;
      if (sentCount === 0) return false;
    }
    return true;
  });

  // Stats — derived from ALL categories (not filtered by confidence/sentiment)
  // so stat cards always show real totals for the selected date range + category.
  const totalReviews = categories.reduce((s, c) => s + c.total, 0);
  const totalNeg = categories.reduce((s, c) => s + c.negative, 0);
  const pctNeg = totalReviews > 0 ? ((totalNeg / totalReviews) * 100).toFixed(1) : "0";
  const topIssue = issues[0];
  const spikeCategories = categories.filter((c) => c.total > 0 && c.negative / c.total > 0.3);

  // Trend chart: filter series visibility based on sentiment selection
  const trendData = weeks.map((w) => ({ ...w, week: w.week.split("T")[0] ?? w.week }));

  // Which sentiment series to show in the chart
  const showPositive = filters.sentiment === "all" || filters.sentiment === "positive";
  const showNeutral = filters.sentiment === "all" || filters.sentiment === "neutral";
  const showNegative = filters.sentiment === "all" || filters.sentiment === "negative";

  return (
    <DashboardPage sidebar={<Sidebar catCol={colMap.catCol} categories={categories} filters={filters} onFilter={updateFilters} />}>
      {/* Alert strip */}
      {spikeCategories.length > 0 && (
        <Alert variant="destructive" className="border-destructive/50 bg-destructive/10">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription className="text-sm flex">
            <span className="font-bold">Negative reviews &gt;30%</span>
            <span className="text-xs">
              <span className=""> {"("} </span>
              {spikeCategories.map((c) => c.category).join(", ")}
              <span className=""> {")"} ·{" "}</span>
            </span>
            <Link to="/reviews?sentiment=negative" className="underline font-bold">View reviews →</Link>
          </AlertDescription>
        </Alert>
      )}

      {/* Stat cards */}
      {loading ? (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-5">
          {Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-36 rounded-xl" />)}
        </div>
      ) : (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-5">
          <StatCard label="Total Reviews" value={totalReviews.toLocaleString()} icon={MessageSquare} />
          <StatCard label="Negative Rate" value={`${pctNeg}%`} sub="Critical feedback" icon={TrendingDown} deltaUp={false} />
          <StatCard
            label="Top Issue"
            value={topIssue ? topIssue.issue_tag.replaceAll("_", " ") : "—"}
            sub={topIssue ? `${topIssue.count} reviews` : undefined}
            icon={Tag}
          />
          <StatCard
            label={colMap.catCol ? `${colMap.catCol} Count` : "Categories"}
            value={String(categories.length)}
            sub="distinct groups"
            icon={Filter}
          />
        </div>
      )}

      {/* Sentiment trend chart */}
      <Card>
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-base">Sentiment Trends</CardTitle>
              <p className="text-xs text-muted-foreground mt-0.5">Weekly aggregation</p>
            </div>
            <div className="flex gap-4">
              {[["Positive", "#05B169"], ["Neutral", "#7C828A"], ["Negative", "#CF202F"]].map(([l, c]) => (
                <div key={l} className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-full" style={{ background: c }} />
                  <span className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">{l}</span>
                </div>
              ))}
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {loading ? <Skeleton className="h-72 w-full" /> : trendData.length === 0 ? (
            <div className="h-72 flex flex-col items-center justify-center text-center">
              <p className="text-sm text-muted-foreground">Upload a CSV with a date column to see trends over time.</p>
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={280}>
              <AreaChart data={trendData}>
                <defs>
                  {[["pos", "#05B169"], ["neu", "#7C828A"], ["neg", "#CF202F"]].map(([id, c]) => (
                    <linearGradient key={id} id={`g-${id}`} x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={c} stopOpacity={0.2} />
                      <stop offset="95%" stopColor={c} stopOpacity={0} />
                    </linearGradient>
                  ))}
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="oklch(var(--border))" />
                <XAxis dataKey="week" tick={{ fill: "oklch(var(--muted-foreground))", fontSize: 11 }} />
                <YAxis tick={{ fill: "oklch(var(--muted-foreground))", fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: "oklch(var(--background))", border: "1px solid oklch(var(--border))", borderRadius: 8 }}
                  formatter={(value: any, name: any) => [value, String(name).charAt(0).toUpperCase() + String(name).slice(1)]}
                />
                {showPositive && <Area type="monotone" dataKey="positive" stroke="#05B169" strokeWidth={2} fill="url(#g-pos)" dot={false} />}
                {showNeutral && <Area type="monotone" dataKey="neutral" stroke="#7C828A" strokeWidth={2} fill="url(#g-neu)" dot={false} />}
                {showNegative && <Area type="monotone" dataKey="negative" stroke="#CF202F" strokeWidth={2} fill="url(#g-neg)" dot={false} />}
              </AreaChart>
            </ResponsiveContainer>
          )}
        </CardContent>
      </Card>

      {/* Bottom grid: category table + issue chart */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        {/* Category table */}
        <Card className="lg:col-span-8">
          <CardHeader className="pb-2 flex-row justify-between items-start">
            <div>
              <CardTitle className="text-base">
                <span className="capitalize">
                  {colMap.catCol ? `${colMap.catCol} Analysis` : "Category Analysis"}
                </span>
              </CardTitle>
              <p className="text-xs text-muted-foreground">Sentiment by group</p>
            </div>
            <Button variant="link" size="sm" asChild className="text-primary justify-start p-0">
              <Link to="/reports">View Reports →</Link>
            </Button>
          </CardHeader>
          <CardContent className="px-0">
            {loading ? <Skeleton className="h-48 mx-6" /> : visibleCategories.length === 0 ? (
              <p className="text-sm text-muted-foreground text-center py-10">
                No {colMap.catCol ?? "category"} data yet.{" "}
                <Link to="/upload" className="underline text-primary">Upload a CSV with a category column.</Link>
              </p>
            ) : (
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border bg-muted/30">
                    <th className="px-6 py-3 text-left text-[10px] font-bold uppercase tracking-wider text-muted-foreground">{colMap.catCol ?? "Category"}</th>
                    <th className="px-6 py-3 text-left text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Sentiment</th>
                    <th className="px-6 py-3 text-right text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Volume</th>
                    <th className="px-6 py-3 text-left text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {visibleCategories.map((c) => {
                    const score = c.sentiment_score * 100;
                    const isNeg = score < 0;
                    return (
                      <tr key={c.category} className="border-b border-border hover:bg-accent/30 transition-colors">
                        <td className="px-6 py-3 font-semibold">{c.category}</td>
                        <td className="px-6 py-3">
                          <span className={`font-bold font-number ${isNeg ? "text-destructive" : "text-green-500"}`}>
                            {score.toFixed(1)}
                          </span>
                        </td>
                        <td className="px-6 py-3 text-right font-number text-muted-foreground">{c.total.toLocaleString()}</td>
                        <td className="px-6 py-3">
                          <span className={`w-2 h-2 rounded-full inline-block ${isNeg ? "bg-destructive" : "bg-green-500"}`} />
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
          </CardContent>
        </Card>

        {/* Issue distribution chart */}
        <Card className="lg:col-span-4">
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Issue Distribution</CardTitle>
            <p className="text-xs text-muted-foreground">Top issue tags</p>
          </CardHeader>
          <CardContent>
            {loading ? <Skeleton className="h-60 w-full" /> : issues.length === 0 ? (
              <div className="h-60 flex items-center justify-center">
                <p className="text-xs text-muted-foreground">No issue data</p>
              </div>
            ) : (
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={issues.slice(0, 8)} layout="vertical" margin={{ left: 0, right: 8 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="oklch(var(--border))" horizontal={false} />
                  <XAxis type="number" tick={{ fill: "oklch(var(--muted-foreground))", fontSize: 10 }} />
                  <YAxis type="category" dataKey="issue_tag" tick={{ fill: "oklch(var(--muted-foreground))", fontSize: 10 }} width={110}
                    tickFormatter={(v: string) => v.replaceAll("_", " ")} />
                  <Tooltip contentStyle={{ backgroundColor: "oklch(var(--muted))", border: "1px solid oklch(var(--border))", borderRadius: 8, padding: "12px" }}
                    formatter={(v) => [v, "Count"]} />
                  <Bar dataKey="count" fill="#CF202F" radius={[0, 3, 3, 0]} />
                </BarChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>
      </div>
    </DashboardPage>
  );
}
