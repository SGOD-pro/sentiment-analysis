/**
 * Reviews page — matches Stitch "SentiMetric | Reviews (v2)" screen.
 * Card-based feed, sidebar filters, search bar, pagination.
 * All column labels from loadColumnMap() — never hardcoded.
 * All API calls scoped to batchId from useSessionStore.
 */
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Separator } from "@/components/ui/separator";
import { Slider } from "@/components/ui/slider";
import { ChevronDown, ChevronRight, Clock, Filter, Search, Upload } from "lucide-react";
import { getCategoriesSummary, getIssuesDistribution, getReviews, correctReview } from "@/api/client";
import { loadColumnMap } from "@/hooks/useColumnMap";
import { useSessionStore } from "@/hooks/useSessionStore";
import type { Correction, CategorySummary, IssueCount, Review, ReviewFilters } from "@/types";
import { DashboardPage } from "@/components/dashboard-layout";
import { DateRangeFilter, type DateRangeValue } from "@/components/DateRangeFilter";
import { cn } from "@/lib/utils";

const SENT_STYLE: Record<string, string> = {
  positive: "chip-positive",
  neutral: "chip-neutral",
  negative: "chip-negative",
};

const SENT_LABELS = ["positive", "neutral", "negative"] as const;
type Sentiment = typeof SENT_LABELS[number];

function CorrectionPanel({ review, onCorrect }: {
  review: Review;
  onCorrect: (reviewId: string, label: string, correction: Correction) => void;
}) {
  const [busy, setBusy] = useState(false);
  const [pendingLabel, setPendingLabel] = useState<Sentiment | null>(null);
  const current = review.correction?.manual_label ?? review.sentiment;
  const original = review.sentiment;

  function handleCorrectClick(label: Sentiment) {
    if (label === current || busy) return;
    setPendingLabel(label);
  }

  async function confirmCorrection() {
    if (!pendingLabel || busy) return;
    setBusy(true);
    const labelToSave = pendingLabel;
    
    try {
      const res = await correctReview(review.review_id, labelToSave);
      if (!res.success || !res.data) {
        toast.error(res.message ?? "Failed to save correction");
        return;
      }
      toast.success(`Marked as ${labelToSave}`);
      onCorrect(review.review_id, labelToSave, res.data);
    } catch {
      toast.error("Network error saving correction");
    } finally {
      setBusy(false);
      setPendingLabel(null);
    }
  }

  return (
    <div className="space-y-2">
      <p className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
        {review.correction ? "Correction saved" : "Correct this prediction"}
      </p>
      <div className="flex gap-2">
        {SENT_LABELS.map((label) => (
          <button
            key={label}
            disabled={busy || label === current}
            onClick={() => handleCorrectClick(label)}
            className={`text-xs px-2.5 py-1 rounded-sm font-bold transition-all border ${
              label === current
                ? `${SENT_STYLE[label]} opacity-60 cursor-default`
                : "border-border text-muted-foreground hover:text-foreground hover:border-primary"
            }`}
          >
            {label.toUpperCase()}
          </button>
        ))}
      </div>
      {review.correction && (
        <p className="text-[10px] text-muted-foreground">
          Was: <span className={`font-bold ${SENT_STYLE[original]}`}>{original.toUpperCase()}</span>
          {" → "}
          Now: <span className={`font-bold ${SENT_STYLE[review.correction.manual_label]}`}>{review.correction.manual_label.toUpperCase()}</span>
        </p>
      )}

      {/* Confirmation Dialog */}
      <Dialog open={pendingLabel !== null} onOpenChange={(open) => { if (!open && !busy) setPendingLabel(null); }}>
        <DialogContent className="w-80">
          <DialogHeader>
            <DialogTitle>Confirm Correction</DialogTitle>
            <DialogDescription>
              Are you sure you want to manually override the sentiment prediction to{" "}
              <strong className={cn(pendingLabel ? SENT_STYLE[pendingLabel] : "", "font-bold text-xs p-1 rounded")}>{pendingLabel?.toUpperCase()}</strong>?
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="mt-4">
            <Button variant="outline" onClick={() => setPendingLabel(null)} disabled={busy}>
              Cancel
            </Button>
            <Button onClick={confirmCorrection} disabled={busy}>
              {busy ? "Saving..." : "Confirm"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

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
    <Card onClick={onClick} className={`cursor-pointer hover:shadow-md transition-shadow duration-200 border-border ${review.sentiment === "positive" ? "bg-green-500/5" :
        review.sentiment === "negative" ? "bg-destructive/5" : "bg-muted/20"
      }`}>
      <CardContent className="">
        {/* Top row */}
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2 flex-wrap">
            <span className={`text-xs font-bold px-2 py-0.5 rounded-sm ${SENT_STYLE[review.sentiment]}`}>
              {review.sentiment.toUpperCase()}
            </span>
            {review.correction && (
              <span className={`text-xs font-bold px-2 py-0.5 rounded-sm border border-dashed ${SENT_STYLE[review.correction.manual_label]}`}
                title={`Corrected: ${review.sentiment} → ${review.correction.manual_label}`}>
                ✏ {review.correction.manual_label.toUpperCase()}
              </span>
            )}
            {review.category && (
              <Badge variant="secondary" className="text-xs font-normal">{review.category}</Badge>
            )}
            {review.review_date && (
              <span className="text-xs text-muted-foreground flex items-center gap-1"><Clock size={10} className="mb-0.5" /> {review.review_date}</span>
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
  const batchId = useSessionStore((s) => s.batchId);
  const navigate = useNavigate();
  const [reviews, setReviews] = useState<Review[]>([]);
  const [total, setTotal] = useState(0);
  const [totalPages, setTotalPages] = useState(1);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<Review | null>(null);

  // Optimistic correction map: review_id -> Correction
  const [corrections, setCorrections] = useState<Record<string, Correction>>({});

  function handleCorrect(reviewId: string, _label: string, correction: Correction) {
    setCorrections((prev) => ({ ...prev, [reviewId]: correction }));
    // Also update the selected dialog if it's the same review
    setSelected((prev) => prev && prev.review_id === reviewId ? { ...prev, correction } : prev);
  }

  // Filter state
  const [sentiment, setSentiment] = useState("all");
  const [category, setCategory] = useState("all");
  const [issueTag, setIssueTag] = useState("all");
  const [minConf, setMinConf] = useState(0);
  const [dateRange, setDateRange] = useState<DateRangeValue>({});

  // Option lists
  const [catOptions, setCatOptions] = useState<CategorySummary[]>([]);
  const [issueOptions, setIssueOptions] = useState<IssueCount[]>([]);

  useEffect(() => {
    if (!batchId) { setLoading(false); return; }
    getCategoriesSummary(batchId).then((r) => { if (r.data) setCatOptions(r.data.categories); });
    getIssuesDistribution(batchId).then((r) => { if (r.data) setIssueOptions(r.data.issues); });
  }, [batchId]);

  useEffect(() => {
    if (!batchId) { setLoading(false); return; }
    setLoading(true);
    const f: ReviewFilters = { page, limit: 25 };
    if (sentiment !== "all") f.sentiment = sentiment;
    if (category !== "all") f.category = category;
    if (issueTag !== "all") f.issue_tag = issueTag;
    if (dateRange.from) f.from = dateRange.from;
    if (dateRange.to) f.to = dateRange.to;

    getReviews(batchId, f).then((r) => {
      if (!r.success) { toast.error(r.message ?? "Failed to load reviews"); return; }
      setReviews(r.data?.reviews ?? []);
      setTotal(r.data?.total ?? 0);
      setTotalPages(r.data?.total_pages ?? 1);
    }).finally(() => setLoading(false));
  }, [batchId, page, sentiment, category, issueTag, minConf, dateRange]);

  if (!batchId) {
    return (
      <DashboardPage sidebar={<div className="p-5"><p className="text-xs text-muted-foreground">No active session</p></div>}>
        <div className="flex-1 flex flex-col items-center justify-center gap-4 p-10 text-center">
          <Upload className="w-16 h-16 text-muted-foreground/30" />
          <h2 className="text-xl font-semibold">No data loaded</h2>
          <p className="text-sm text-muted-foreground">Upload a CSV to browse reviews.</p>
          <Button onClick={() => navigate("/upload")} className="gap-2"><Upload className="w-4 h-4" />Import Data</Button>
        </div>
      </DashboardPage>
    );
  }

  const filtered = minConf > 0
    ? reviews.filter((r) => Number(r.confidence_margin) >= minConf)
    : reviews;

  const pageNums = Array.from({ length: Math.min(totalPages, 5) }, (_, i) => i + 1);

  const reviewsSidebar = (
    <div className="flex flex-col p-5 gap-2 h-full">
      <div className="flex items-center gap-2 pb-1">
        <Filter className="w-4 h-4 text-primary" />
        <span className="font-semibold text-sm">Filters</span>
      </div>
      <p className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground">Analysis Parameters</p>
      <Separator />

      <div className="flex-1 overflow-y-auto space-y-1 pr-1">
        <FilterSection label="Date Range" defaultOpen>
          <DateRangeFilter onChange={(v) => { setDateRange(v); setPage(1); }} />
        </FilterSection>
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
    </div>
  );

  return (
    <DashboardPage sidebar={reviewsSidebar}>
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
              <ReviewCard
                key={r.review_id}
                review={{ ...r, correction: corrections[r.review_id] ?? r.correction }}
                colMap={colMap}
                onClick={() => setSelected({ ...r, correction: corrections[r.review_id] ?? r.correction })}
              />
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

      {/* Detail dialog */}
      <Dialog open={!!selected} onOpenChange={(o) => !o && setSelected(null)}>
        <DialogContent className="sm:max-w-2xl">
          {selected && (
            <div className="space-y-4">
              <div className="flex items-center gap-2 flex-wrap">
                <span className={`text-xs font-bold px-2 py-0.5 rounded-sm ${SENT_STYLE[selected.sentiment]}`}>
                  {selected.sentiment.toUpperCase()}
                </span>
                {selected.correction && <Badge variant="secondary" className="bg-amber-100 text-amber-800">Corrected</Badge>}
                {selected.issue_tag && <Badge variant="outline">{selected.issue_tag.replaceAll("_", " ")}</Badge>}
                {selected.category && <Badge variant="secondary">{selected.category}</Badge>}
              </div>
              <p className="text-sm leading-relaxed">"{selected.text}"</p>
              <Separator />
              <div className="grid grid-cols-2 gap-3 text-xs">
                <div><p className="text-muted-foreground font-bold uppercase mb-1">Confidence</p><p className="font-number">{Number(selected.confidence_margin).toFixed(4)}</p></div>
                <div><p className="text-muted-foreground font-bold uppercase mb-1">Probabilities</p>
                  <p className="font-number">P:{Number(selected.prob_positive).toFixed(3)} Neu:{Number(selected.prob_neutral).toFixed(3)} Neg:{Number(selected.prob_negative).toFixed(3)}</p>
                </div>
                {selected.review_date && <div><p className="text-muted-foreground font-bold uppercase mb-1">Date</p><p className="font-number">{selected.review_date}</p></div>}
                {colMap.extraCols.map((col) => (
                  <div key={col}><p className="text-muted-foreground font-bold uppercase mb-1">{col}</p><p>{String(selected[col] ?? "—")}</p></div>
                ))}
              </div>
              <Separator />
              <CorrectionPanel review={selected} onCorrect={handleCorrect} />
            </div>
          )}
        </DialogContent>
      </Dialog>
    </DashboardPage>
  );
}
