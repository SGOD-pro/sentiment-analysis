/**
 * Reviews page — matches Stitch "SentiMetric | Reviews (v2)" screen.
 * Card-based feed, sidebar filters, search bar, pagination.
 * All column labels from loadColumnMap() — never hardcoded.
 */
import { useEffect, useState } from "react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Separator } from "@/components/ui/separator";
import { Slider } from "@/components/ui/slider";
import { ChevronDown, ChevronRight, Filter, Search } from "lucide-react";
import { getCategoriesSummary, getIssuesDistribution, getReviews } from "@/api/client";
import { loadColumnMap } from "@/hooks/useColumnMap";
import type { CategorySummary, IssueCount, Review, ReviewFilters } from "@/types";

const SENT_STYLE: Record<string, string> = {
  positive: "chip-positive",
  neutral: "chip-neutral",
  negative: "chip-negative",
};

function FilterSection({ label, children, defaultOpen = false }: { label: string; children: React.ReactNode; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div>
      <button className="w-full flex items-center justify-between py-2 text-xs font-medium text-muted-foreground hover:text-foreground"
        onClick={() => setOpen((v) => !v)}>
        {label} {open ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
      </button>
      {open && <div className="pb-3">{children}</div>}
    </div>
  );
}

function ReviewCard({ review, colMap, onClick }: {
  review: Review;
  colMap: ReturnType<typeof loadColumnMap>;
  onClick: () => void;
}) {
  return (
    <Card onClick={onClick} className={`cursor-pointer hover:shadow-md transition-shadow duration-200 border-border ${
      review.sentiment === "positive" ? "bg-green-500/5" :
      review.sentiment === "negative" ? "bg-destructive/5" : "bg-muted/20"
    }`}>
      <CardContent className="pt-4 pb-4">
        {/* Top row */}
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2 flex-wrap">
            <span className={`text-xs font-bold px-2 py-0.5 rounded-sm ${SENT_STYLE[review.sentiment]}`}>
              {review.sentiment.toUpperCase()}
            </span>
            {review.category && (
              <Badge variant="secondary" className="text-xs font-normal">{review.category}</Badge>
            )}
            {review.review_date && (
              <span className="text-xs text-muted-foreground flex items-center gap-1">🕐 {review.review_date}</span>
            )}
          </div>
          <div className="text-right">
            <p className="text-[10px] text-muted-foreground uppercase font-bold">AI Confidence</p>
            <p className="font-number text-sm font-bold text-primary">{Number(review.confidence_margin).toFixed(3)}</p>
          </div>
        </div>
        {/* Review text */}
        <p className="text-sm leading-relaxed line-clamp-3">"{review[colMap.textCol] as string ?? review.text}"</p>
        {/* Footer: issue tags + extra cols */}
        <div className="flex items-center justify-between mt-3 flex-wrap gap-2">
          <div className="flex gap-1.5 flex-wrap">
            {review.issue_tag && (
              <Badge variant="outline" className="text-[10px] font-number">{review.issue_tag.replaceAll("_", " ")}</Badge>
            )}
          </div>
          <div className="flex gap-3">
            {colMap.extraCols.map((col) => (
              <span key={col} className="text-xs text-muted-foreground">{col}: <span className="font-medium">{String(review[col] ?? "—")}</span></span>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export default function Reviews() {
  const colMap = loadColumnMap();
  const [reviews, setReviews] = useState<Review[]>([]);
  const [total, setTotal] = useState(0);
  const [totalPages, setTotalPages] = useState(1);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<Review | null>(null);

  // Filter state
  const [sentiment, setSentiment] = useState("all");
  const [category, setCategory] = useState("all");
  const [issueTag, setIssueTag] = useState("all");
  const [minConf, setMinConf] = useState(0);

  // Option lists
  const [catOptions, setCatOptions] = useState<CategorySummary[]>([]);
  const [issueOptions, setIssueOptions] = useState<IssueCount[]>([]);

  useEffect(() => {
    getCategoriesSummary().then((r) => { if (r.data) setCatOptions(r.data.categories); });
    getIssuesDistribution().then((r) => { if (r.data) setIssueOptions(r.data.issues); });
  }, []);

  useEffect(() => {
    setLoading(true);
    const f: ReviewFilters = { page, limit: 25 };
    if (sentiment !== "all") f.sentiment = sentiment;
    if (category !== "all") f.category = category;
    if (issueTag !== "all") f.issue_tag = issueTag;

    getReviews(f).then((r) => {
      if (!r.success) { toast.error(r.message ?? "Failed to load reviews"); return; }
      setReviews(r.data?.reviews ?? []);
      setTotal(r.data?.total ?? 0);
      setTotalPages(r.data?.total_pages ?? 1);
    }).finally(() => setLoading(false));
  }, [page, sentiment, category, issueTag, minConf]);

  const filtered = minConf > 0
    ? reviews.filter((r) => Number(r.confidence_margin) >= minConf)
    : reviews;

  const pageNums = Array.from({ length: Math.min(totalPages, 5) }, (_, i) => i + 1);

  return (
    <div className="flex pt-14 min-h-screen">
      {/* Sidebar */}
      <aside className="fixed left-0 top-14 h-[calc(100vh-56px)] w-60 border-r border-border bg-card flex flex-col p-5 gap-2 z-40 hidden md:flex">
        <div className="flex items-center gap-2 pb-1">
          <Filter className="w-4 h-4 text-primary" />
          <span className="font-semibold text-sm">Filters</span>
        </div>
        <p className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground">Analysis Parameters</p>
        <Separator />

        <div className="flex-1 overflow-y-auto space-y-1 pr-1">
          <FilterSection label="Date Range"><p className="text-xs text-muted-foreground">Coming soon</p></FilterSection>
          {colMap.catCol && (
            <FilterSection label={colMap.catCol} defaultOpen>
              <Select value={category} onValueChange={(v) => { setCategory(v); setPage(1); }}>
                <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All</SelectItem>
                  {catOptions.map((c) => <SelectItem key={c.category} value={c.category}>{c.category}</SelectItem>)}
                </SelectContent>
              </Select>
            </FilterSection>
          )}
          <FilterSection label="Sentiment" defaultOpen>
            <Select value={sentiment} onValueChange={(v) => { setSentiment(v); setPage(1); }}>
              <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All</SelectItem>
                <SelectItem value="positive">Positive</SelectItem>
                <SelectItem value="neutral">Neutral</SelectItem>
                <SelectItem value="negative">Negative</SelectItem>
              </SelectContent>
            </Select>
          </FilterSection>
          <FilterSection label="Issue Tag">
            <Select value={issueTag} onValueChange={(v) => { setIssueTag(v); setPage(1); }}>
              <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="all">Any</SelectItem>
                {issueOptions.map((i) => <SelectItem key={i.issue_tag} value={i.issue_tag}>{i.issue_tag.replaceAll("_", " ")}</SelectItem>)}
              </SelectContent>
            </Select>
          </FilterSection>
          <FilterSection label="Confidence">
            <Slider min={0} max={1} step={0.05} value={[minConf]} onValueChange={([v]) => setMinConf(v)} className="w-full" />
            <div className="flex justify-between text-[10px] text-muted-foreground mt-1">
              <span>0%</span><span>{Math.round(minConf * 100)}%</span>
            </div>
          </FilterSection>
        </div>

        <Button className="w-full mt-auto" onClick={() => setPage(1)}>Apply Filters</Button>
      </aside>

      {/* Main */}
      <main className="md:pl-60 flex-1 p-6">
        {/* Page header */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 mb-6">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">Review Feed</h1>
            <p className="text-sm text-muted-foreground">Analyze real-time sentiment data</p>
          </div>
          <div className="flex gap-2">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <Input className="pl-9 w-56 h-9 text-sm" placeholder="Search reviews…" readOnly title="Search coming soon" />
            </div>
            <Select defaultValue="recent">
              <SelectTrigger className="h-9 w-36 text-sm"><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="recent">Most Recent</SelectItem>
                <SelectItem value="confidence">By Confidence</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        {/* Review cards */}
        <div className="space-y-4">
          {loading
            ? Array.from({ length: 5 }).map((_, i) => <Skeleton key={i} className="h-36 w-full rounded-xl" />)
            : filtered.length === 0
            ? <p className="text-center text-muted-foreground py-16">No reviews match your filters.</p>
            : filtered.map((r) => (
                <ReviewCard key={r.review_id} review={r} colMap={colMap} onClick={() => setSelected(r)} />
              ))}
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between mt-6">
            <p className="text-xs text-muted-foreground">Showing {((page - 1) * 25) + 1}–{Math.min(page * 25, total)} of {total.toLocaleString()} reviews</p>
            <div className="flex gap-1">
              <Button variant="outline" size="sm" disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>‹</Button>
              {pageNums.map((n) => (
                <Button key={n} variant={n === page ? "default" : "outline"} size="sm" onClick={() => setPage(n)}>{n}</Button>
              ))}
              {totalPages > 5 && <><span className="px-2 py-1 text-sm text-muted-foreground">…</span><Button variant="outline" size="sm" onClick={() => setPage(totalPages)}>{totalPages}</Button></>}
              <Button variant="outline" size="sm" disabled={page >= totalPages} onClick={() => setPage((p) => p + 1)}>›</Button>
            </div>
          </div>
        )}
      </main>

      {/* Detail dialog */}
      <Dialog open={!!selected} onOpenChange={(o) => !o && setSelected(null)}>
        <DialogContent className="sm:max-w-2xl">
          {selected && (
            <div className="space-y-4">
              <div className="flex items-center gap-2 flex-wrap">
                <span className={`text-xs font-bold px-2 py-0.5 rounded-sm ${SENT_STYLE[selected.sentiment]}`}>
                  {selected.sentiment.toUpperCase()}
                </span>
                {selected.issue_tag && <Badge variant="outline">{selected.issue_tag.replaceAll("_", " ")}</Badge>}
                {selected.category && <Badge variant="secondary">{selected.category}</Badge>}
              </div>
              <p className="text-sm leading-relaxed">"{selected.text}"</p>
              <Separator />
              <div className="grid grid-cols-2 gap-3 text-xs">
                <div><p className="text-muted-foreground font-bold uppercase mb-1">Confidence</p><p className="font-number">{Number(selected.confidence_margin).toFixed(4)}</p></div>
                <div><p className="text-muted-foreground font-bold uppercase mb-1">Probabilities</p>
                  <p className="font-number">P:{Number(selected.prob_positive).toFixed(3)} N:{Number(selected.prob_neutral).toFixed(3)} Ng:{Number(selected.prob_negative).toFixed(3)}</p>
                </div>
                {selected.review_date && <div><p className="text-muted-foreground font-bold uppercase mb-1">Date</p><p className="font-number">{selected.review_date}</p></div>}
                {colMap.extraCols.map((col) => (
                  <div key={col}><p className="text-muted-foreground font-bold uppercase mb-1">{col}</p><p>{String(selected[col] ?? "—")}</p></div>
                ))}
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
