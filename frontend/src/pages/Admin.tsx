/**
 * Admin Corrections Panel — /admin
 *
 * TODO(auth): This page has NO authentication in v1.
 *             Add a token/session guard before any public or multi-tenant deployment.
 *             Access is by direct URL only and deliberately omitted from the main nav.
 */

import { useEffect, useMemo, useState } from "react";
import { ArrowUpDown, Download, ChevronDown, ChevronUp, Lock } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

const BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

interface AdminCorrection {
  correction_id: string;
  review_id: string;
  batch_id: string;
  text: string;
  label: string;         // original model label
  manual_label: string;  // human correction
  date: string;
  category?: string;
  confidence_margin?: string;
}

type SortKey = "date" | "label" | "manual_label";
type SortDir = "asc" | "desc";

const SENTIMENT_COLORS: Record<string, string> = {
  positive: "bg-emerald-500/15 text-emerald-400 border border-emerald-500/30",
  neutral:  "bg-sky-500/15 text-sky-400 border border-sky-500/30",
  negative: "bg-rose-500/15 text-rose-400 border border-rose-500/30",
};

function Badge({ label }: { label: string }) {
  const cls = SENTIMENT_COLORS[label] ?? "bg-zinc-500/15 text-zinc-400 border border-zinc-500/30";
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${cls}`}>
      {label}
    </span>
  );
}

function ExpandableText({ text, limit = 80 }: { text: string; limit?: number }) {
  const [expanded, setExpanded] = useState(false);
  const short = text.length > limit && !expanded;
  return (
    <span
      className="cursor-pointer select-none"
      onClick={() => setExpanded((p) => !p)}
      title={short ? "Click to expand" : undefined}
    >
      {short ? `${text.slice(0, limit)}…` : text}
    </span>
  );
}

function SortButton({
  col, current, dir, onSort,
}: { col: SortKey; current: SortKey; dir: SortDir; onSort: (c: SortKey) => void }) {
  const active = current === col;
  return (
    <button
      onClick={() => onSort(col)}
      className={`flex items-center gap-1 hover:text-foreground transition-colors ${active ? "text-foreground" : "text-muted-foreground"}`}
    >
      {col === "date" ? "Date corrected" : col === "label" ? "Original label" : "Corrected label"}
      {active ? (
        dir === "asc" ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />
      ) : (
        <ArrowUpDown className="w-3 h-3 opacity-50" />
      )}
    </button>
  );
}

function AdminDashboard() {
  const [rows, setRows] = useState<AdminCorrection[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [sortKey, setSortKey] = useState<SortKey>("date");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [filterOriginal, setFilterOriginal] = useState("all");
  const [filterCorrected, setFilterCorrected] = useState("all");

  const [totalCorrections, setTotalCorrections] = useState(0);
  const [batchCount, setBatchCount] = useState(0);

  useEffect(() => {
    setLoading(true);
    fetch(`${BASE}/api/admin/corrections`)
      .then((r) => r.json())
      .then((res) => {
        if (!res.success) throw new Error(res.message ?? "Failed to load corrections");
        setRows(res.data.corrections ?? []);
        setTotalCorrections(res.data.total ?? 0);
        setBatchCount(res.data.batch_count ?? 0);
      })
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const handleSort = (col: SortKey) => {
    if (sortKey === col) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else { setSortKey(col); setSortDir("asc"); }
  };

  const displayed = useMemo(() => {
    let data = rows;
    if (filterOriginal !== "all") data = data.filter((r) => r.label === filterOriginal);
    if (filterCorrected !== "all") data = data.filter((r) => r.manual_label === filterCorrected);
    return [...data].sort((a, b) => {
      let va = sortKey === "date" ? a.date : a[sortKey];
      let vb = sortKey === "date" ? b.date : b[sortKey];
      va = va ?? "";
      vb = vb ?? "";
      return sortDir === "asc" ? va.localeCompare(vb) : vb.localeCompare(va);
    });
  }, [rows, filterOriginal, filterCorrected, sortKey, sortDir]);

  const handleExport = () => {
    window.location.href = `${BASE}/api/admin/corrections?format=csv`;
  };

  return (
    <div className="min-h-screen bg-background text-foreground p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-2xl font-bold tracking-tight">Admin — Corrections</h1>
            <p className="text-muted-foreground text-sm mt-1">
              Human label corrections collected for ML retraining.{" "}
              <span className="text-amber-400 font-medium">No auth (v1)</span>
            </p>
          </div>
          <Button onClick={handleExport} variant="outline" size="sm" className="gap-1.5">
            <Download className="w-4 h-4" />
            Export CSV
          </Button>
        </div>

        {/* Summary */}
        <div className="rounded-lg border border-border bg-card px-4 py-3 text-sm">
          {loading ? (
            <span className="text-muted-foreground">Loading…</span>
          ) : error ? (
            <span className="text-destructive">{error}</span>
          ) : (
            <span>
              <span className="font-semibold text-foreground">{totalCorrections}</span>
              {" "}corrections collected across{" "}
              <span className="font-semibold text-foreground">{batchCount}</span>
              {" "}batch{batchCount !== 1 ? "es" : ""}
            </span>
          )}
        </div>

        {/* Filters */}
        <div className="flex flex-wrap gap-3 items-center">
          <span className="text-sm text-muted-foreground">Filter by:</span>
          <Select value={filterOriginal} onValueChange={setFilterOriginal}>
            <SelectTrigger className="w-44 h-8 text-sm">
              <SelectValue placeholder="Original label" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All original labels</SelectItem>
              <SelectItem value="positive">Positive</SelectItem>
              <SelectItem value="neutral">Neutral</SelectItem>
              <SelectItem value="negative">Negative</SelectItem>
            </SelectContent>
          </Select>
          <Select value={filterCorrected} onValueChange={setFilterCorrected}>
            <SelectTrigger className="w-44 h-8 text-sm">
              <SelectValue placeholder="Corrected label" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All corrected labels</SelectItem>
              <SelectItem value="positive">Positive</SelectItem>
              <SelectItem value="neutral">Neutral</SelectItem>
              <SelectItem value="negative">Negative</SelectItem>
            </SelectContent>
          </Select>
          {(filterOriginal !== "all" || filterCorrected !== "all") && (
            <Button variant="ghost" size="sm" className="h-8 text-xs text-muted-foreground"
              onClick={() => { setFilterOriginal("all"); setFilterCorrected("all"); }}>
              Clear filters
            </Button>
          )}
          {!loading && !error && (filterOriginal !== "all" || filterCorrected !== "all") && (
            <span className="text-xs text-muted-foreground ml-auto">
              {displayed.length} of {totalCorrections} shown
            </span>
          )}
        </div>

        {/* Table */}
        <div className="rounded-lg border border-border overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-muted/40 border-b border-border">
                <tr>
                  <th className="text-left px-4 py-3 font-medium">
                    <SortButton col="date" current={sortKey} dir={sortDir} onSort={handleSort} />
                  </th>
                  <th className="text-left px-4 py-3 font-medium text-muted-foreground">Review text</th>
                  <th className="text-left px-4 py-3 font-medium">
                    <SortButton col="label" current={sortKey} dir={sortDir} onSort={handleSort} />
                  </th>
                  <th className="text-left px-4 py-3 font-medium">
                    <SortButton col="manual_label" current={sortKey} dir={sortDir} onSort={handleSort} />
                  </th>
                  <th className="text-left px-4 py-3 font-medium text-muted-foreground">Category</th>
                  <th className="text-left px-4 py-3 font-medium text-muted-foreground">Batch ID</th>
                  <th className="text-left px-4 py-3 font-medium text-muted-foreground">Review ID</th>
                  <th className="text-left px-4 py-3 font-medium text-muted-foreground">Confidence</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {loading && (
                  <tr>
                    <td colSpan={8} className="px-4 py-8 text-center text-muted-foreground">
                      Loading corrections…
                    </td>
                  </tr>
                )}
                {error && (
                  <tr>
                    <td colSpan={8} className="px-4 py-8 text-center text-destructive">{error}</td>
                  </tr>
                )}
                {!loading && !error && displayed.length === 0 && (
                  <tr>
                    <td colSpan={8} className="px-4 py-8 text-center text-muted-foreground">
                      No corrections found.
                    </td>
                  </tr>
                )}
                {displayed.map((row) => (
                  <tr key={row.correction_id} className="hover:bg-muted/20 transition-colors">
                    <td className="px-4 py-3 whitespace-nowrap text-muted-foreground">
                      {row.date ? new Date(row.date).toLocaleDateString() : "—"}
                    </td>
                    <td className="px-4 py-3 max-w-xs text-foreground">
                      <ExpandableText text={row.text ?? ""} />
                    </td>
                    <td className="px-4 py-3"><Badge label={row.label} /></td>
                    <td className="px-4 py-3"><Badge label={row.manual_label} /></td>
                    <td className="px-4 py-3 text-muted-foreground">{row.category ?? "—"}</td>
                    <td className="px-4 py-3 font-mono text-xs text-muted-foreground truncate max-w-[8rem]" title={row.batch_id}>
                      {row.batch_id}
                    </td>
                    <td className="px-4 py-3 font-mono text-xs text-muted-foreground truncate max-w-[8rem]" title={row.review_id}>
                      {row.review_id}
                    </td>
                    <td className="px-4 py-3 text-muted-foreground">
                      {row.confidence_margin ? Number(row.confidence_margin).toFixed(3) : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        
        {/* ML Retraining Information Panel */}
        <div className="mt-8 rounded-lg border border-border bg-card p-6">
          <h2 className="text-lg font-semibold tracking-tight text-foreground mb-4">ML Pipeline Retraining & Artifacts</h2>
          <div className="text-sm text-muted-foreground space-y-4">
            <p>
              When a sufficient number of corrections are collected, you can trigger an automated retraining of the sentiment model to incorporate human feedback.
            </p>
            
            <div>
              <h3 className="font-medium text-foreground mb-1">Steps to Retrain & Validate</h3>
              <ol className="list-decimal list-inside space-y-1 ml-1">
                <li>Go to the <strong>Actions</strong> tab in GitHub and select the <strong>ML Retrain</strong> workflow.</li>
                <li>Trigger the workflow manually. You can adjust the <code>min_corrections</code> threshold (default is 10).</li>
                <li>The pipeline automatically exports corrections from DynamoDB and merges them into the `v4` training set.</li>
                <li>The MLP is retrained and evaluated against <strong>two strict quality gates</strong>:
                  <ul className="list-disc list-inside ml-5 mt-1 text-xs space-y-0.5">
                    <li><span className="font-medium">Standard Test Set:</span> Negative recall must remain &ge; 0.7915.</li>
                    <li><span className="font-medium">Frozen Eval Slice:</span> Neutral recall must remain &ge; 0.2967 (on the hand-verified 300-row difficult-neutral set).</li>
                  </ul>
                </li>
                <li>If the model passes <strong>both</strong> gates, it automatically uploads the new weights to S3 and triggers the main CI/CD deployment.</li>
              </ol>
            </div>

            <div className="pt-2 border-t border-border/50">
              <h3 className="font-medium text-foreground mb-1">Required S3 Data Artifacts</h3>
              <p className="text-xs mb-2">
                The retraining pipeline requires the core dataset splits to be present in S3 at <code>s3://&lt;bucket&gt;/ml-artifacts/training_data/</code>. 
                Ensure these files are uploaded:
              </p>
              <ul className="list-disc list-inside text-xs space-y-0.5 font-mono">
                <li>bge_clean_embeddings.npy</li>
                <li>bge_clean_metadata.parquet</li>
                <li>clean_train_idx_v4.npy</li>
                <li>clean_test_idx_v4.npy</li>
                <li>difficult_neutral_eval_FROZEN.csv</li>
              </ul>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}

export default function Admin() {
  const [auth, setAuth] = useState(false);
  const [password, setPassword] = useState("");
  const [error, setError] = useState(false);

  // Simple static auth for v1
  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const res = await fetch(`${BASE}/api/admin/auth`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ password }),
      });
      const data = await res.json();
      if (data.success && data.data?.authenticated) {
        setAuth(true);
        setError(false);
      } else {
        setError(true);
        setPassword("");
      }
    } catch (err) {
      setError(true);
      setPassword("");
    }
  };

  if (!auth) {
    return (
      <div className="min-h-screen bg-background flex flex-col items-center justify-center p-6">
        <Card className="w-full max-w-2xl">
          <CardHeader className="text-center pb-4">
            <div className="mx-auto bg-primary/10 w-12 h-12 rounded-full flex items-center justify-center mb-4">
              <Lock className="w-6 h-6 text-primary" />
            </div>
            <CardTitle className="text-2xl">Admin Access</CardTitle>
            <CardDescription>Enter the admin password to continue</CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleLogin} className="space-y-4">
              <div className="space-y-2">
                <Input
                  type="password"
                  placeholder="Password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className={error ? "border-destructive focus-visible:ring-destructive" : ""}
                  autoFocus
                />
                {error && <p className="text-xs text-destructive">Incorrect password</p>}
              </div>
              <Button type="submit" className="w-full">
                Verify
              </Button>
            </form>
          </CardContent>
        </Card>
      </div>
    );
  }

  return <AdminDashboard />;
}

