/**
 * Upload page — matches Stitch "SentiMetric | Data Import & Analysis (v2)" screen.
 * Step 1: Data Source (drag-drop). Step 2: Column Mapping + preview table.
 * Step 3: Processing progress. Saves column map to localStorage.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { CheckCircle2, FileUp, Zap, AlertCircle, ArrowLeft } from "lucide-react";
import { getBatchStatus, uploadCSV } from "@/api/client";
import { saveColumnMap } from "@/hooks/useColumnMap";

type Step = "pick" | "map" | "processing" | "done";

export default function Upload() {
  const navigate = useNavigate();
  const inputRef = useRef<HTMLInputElement>(null);
  const [step, setStep] = useState<Step>("pick");
  const [file, setFile] = useState<File | null>(null);
  const [headers, setHeaders] = useState<string[]>([]);
  const [preview, setPreview] = useState<string[][]>([]);
  const [textCol, setTextCol] = useState("");
  const [dateCol, setDateCol] = useState("");
  const [catCol, setCatCol] = useState("");
  const [batchId, setBatchId] = useState("");
  const [processed, setProcessed] = useState(0);
  const [total, setTotal] = useState(0);
  const [error, setError] = useState("");

  const parseCSV = useCallback((f: File) => {
    if (!f.name.endsWith(".csv")) { toast.error("Please upload a .csv file"); return; }
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target?.result as string;
      const lines = text.split("\n").filter((l) => l.trim());
      if (lines.length < 2) { toast.error("CSV must have at least 2 rows"); return; }
      // Regex splits on commas that are outside of double quotes
      const csvSplit = /,(?=(?:(?:[^"]*"){2})*[^"]*$)/;
      const hdrs = lines[0].split(csvSplit).map((h) => h.trim().replace(/^"|"$/g, ""));
      const rows = lines.slice(1, 6).map((l) => l.split(csvSplit).map((c) => c.trim().replace(/^"|"$/g, "")));
      setHeaders(hdrs);
      setPreview(rows);

      // Auto-detect columns based on lowercase names
      const lowerHdrs = hdrs.map(h => h.toLowerCase());
      const autoText = hdrs.find((_, i) => ["text", "review", "reviews", "sentiment"].includes(lowerHdrs[i])) || hdrs[0];
      const autoDate = hdrs.find((_, i) => ["date", "created_at", "timestamp"].includes(lowerHdrs[i])) || "";
      const autoCat = hdrs.find((_, i) => ["category", "categories", "tag"].includes(lowerHdrs[i])) || "";

      setTextCol(autoText);
      setDateCol(autoDate);
      setCatCol(autoCat);
      setFile(f);
      setStep("map");
    };
    reader.readAsText(f);
  }, []);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f) parseCSV(f);
  }, [parseCSV]);

  const handleSubmit = async () => {
    if (!file || !textCol) return;
    const extraCols = headers.filter((h) => h !== textCol && h !== catCol && h !== dateCol);
    try {
      const res = await uploadCSV(file, textCol, catCol || undefined, dateCol || undefined);
      if (!res.success || !res.data) { setError(res.message ?? "Upload failed"); return; }
      saveColumnMap({ textCol, catCol: catCol || undefined, dateCol: dateCol || undefined, extraCols });
      setBatchId(res.data.batch_id);
      setStep("processing");
      toast.info("Analysis started — processing in the background");
    } catch (err) {
      setError(String(err));
    }
  };

  // Poll batch status
  useEffect(() => {
    if (step !== "processing" || !batchId) return;
    const id = setInterval(async () => {
      try {
        const res = await getBatchStatus(batchId);
        if (!res.data) return;
        setProcessed(Number(res.data.processed_count));
        setTotal(Number(res.data.total_reviews));
        if (res.data.status === "done") {
          setStep("done");
          toast.success(`Analysis complete — ${res.data.total_reviews} reviews processed`);
          clearInterval(id);
        }
        if (res.data.status === "failed") {
          setError("Processing failed on the server");
          clearInterval(id);
        }
      } catch { /* retry */ }
    }, 1500);
    return () => clearInterval(id);
  }, [step, batchId]);

  return (
    <div className="pt-14 min-h-screen">
      <main className="max-w-4xl mx-auto p-6 space-y-4">
        <div className="flex items-center gap-4">
          <Button variant="outline" size="icon" onClick={() => window.history.back()} className="shrink-0 fixed top-4 left-4 md:top-8 md:left-8">
            <ArrowLeft className="w-5 h-5" />
          </Button>
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">Import Sentiment Data</h1>
            <p className="text-sm text-muted-foreground mt-1 max-w-2xl">
              Upload your customer reviews, support tickets, or feedback logs in CSV format to begin technical sentiment decomposition and thematic clustering.
            </p>
          </div>
        </div>

        {error && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {/* Step 1 — Data Source */}
        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <span className="w-7 h-7 rounded-full bg-primary text-primary-foreground text-xs font-bold flex items-center justify-center">1</span>
                <CardTitle className="text-base">Data Source</CardTitle>
              </div>
              {file && <Badge variant="outline" className="text-xs"><FileUp className="w-3 h-3 mr-1" />{file.name}</Badge>}
            </div>
          </CardHeader>
          <CardContent>
            <div
              onDrop={onDrop}
              onDragOver={(e) => e.preventDefault()}
              onClick={() => step === "pick" && inputRef.current?.click()}
              className={`border-2 border-dashed rounded-lg p-10 text-center transition-colors ${step === "pick" ? "cursor-pointer hover:border-primary hover:bg-primary/5" : "border-border bg-muted/20"
                }`}
            >
              <FileUp className="w-10 h-10 mx-auto text-muted-foreground mb-3" />
              {step === "pick" ? (
                <>
                  <p className="text-sm">Drop your CSV here or <span className="text-primary font-semibold cursor-pointer">click to browse</span></p>
                  <p className="text-xs text-muted-foreground mt-1">.csv only. Max 50MB</p>
                </>
              ) : (
                <p className="text-sm text-muted-foreground">File loaded — see column mapping below</p>
              )}
              <input ref={inputRef} type="file" accept=".csv" className="hidden" onChange={(e) => { const f = e.target.files?.[0]; if (f) parseCSV(f); }} />
            </div>
          </CardContent>
        </Card>

        {/* Step 2 — Column Mapping + preview */}
        {(step === "map" || step === "processing" || step === "done") && (
          <Card>
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <span className="w-7 h-7 rounded-full bg-primary text-primary-foreground text-xs font-bold flex items-center justify-center">2</span>
                  <CardTitle className="text-base">Column Mapping</CardTitle>
                </div>
                {step === "map" && <Badge variant="destructive" className="text-[10px]">REQUIRED ACTION</Badge>}
              </div>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 sm:grid-cols-3 md:grid-cols-4 gap-6">
                {/* Preview table */}
                <div className="sm:col-span-2 md:col-span-3">
                  <div className="flex items-center justify-between mb-2">
                    <p className="text-sm font-semibold">Data Preview</p>
                    <p className="text-xs text-muted-foreground">Showing first {preview.length} rows</p>
                  </div>
                  <div className="overflow-x-auto rounded-lg border border-border">
                    <table className="w-full text-xs">
                      <thead><tr className="border-b border-border bg-muted/40">
                        {headers.map((h) => <th key={h} className="px-3 py-2 text-left font-semibold text-muted-foreground">{h}</th>)}
                      </tr></thead>
                      <tbody>
                        {preview.map((row, i) => (
                          <tr key={i} className="border-b border-border last:border-0">
                            {row.map((cell, j) => <td key={j} className="px-3 py-2 max-w-[140px] truncate">{cell}</td>)}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Selectors */}
                <div className="space-y-4">
                  <div>
                    <label className="text-xs font-bold uppercase tracking-wider text-muted-foreground">Text Column <span className="text-destructive">*</span></label>
                    <Select value={textCol} onValueChange={setTextCol} disabled={step !== "map"}>
                      <SelectTrigger className="mt-1.5 w-full"><SelectValue /></SelectTrigger>
                      <SelectContent>{headers.map((h) => <SelectItem key={h} value={h}>{h}</SelectItem>)}</SelectContent>
                    </Select>
                    <p className="text-xs text-muted-foreground mt-1">Analyzed for sentiment.</p>
                  </div>
                  <div>
                    <label className="text-xs font-bold uppercase tracking-wider text-muted-foreground">Date Column <span className="ext-muted-foreground/80 text-[10px]">(optional)</span></label>
                    <Select value={dateCol || "__none"} onValueChange={(v) => setDateCol(v === "__none" ? "" : v)} disabled={step !== "map"}>
                      <SelectTrigger className="mt-1.5 w-full"><SelectValue placeholder="— none —" /></SelectTrigger>
                      <SelectContent>
                        <SelectItem value="__none">— none —</SelectItem>
                        {headers.filter((h) => h !== textCol).map((h) => <SelectItem key={h} value={h}>{h}</SelectItem>)}
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <label className="text-xs font-bold uppercase tracking-wider text-muted-foreground">Category Column <span className="text-muted-foreground/80 text-[10px]">(optional)</span></label>
                    <Select value={catCol || "__none"} onValueChange={(v) => setCatCol(v === "__none" ? "" : v)} disabled={step !== "map"}>
                      <SelectTrigger className="mt-1.5 w-full"><SelectValue placeholder="— none —" /></SelectTrigger>
                      <SelectContent>
                        <SelectItem value="__none">— none —</SelectItem>
                        {headers.filter((h) => h !== textCol && h !== dateCol).map((h) => <SelectItem key={h} value={h}>{h}</SelectItem>)}
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Step 3 — Submit / processing / done */}
        {step === "map" && (
          <div className="flex items-center justify-between p-5 rounded-xl border border-border bg-card">
            <div>
              <p className="text-2xl font-bold font-number">{preview.length.toLocaleString()} Records</p>
              <p className="text-xs text-muted-foreground">Estimated processing time: ~{Math.round(preview.length * 0.01 + 10)} seconds</p>
            </div>
            <Button size="lg" onClick={handleSubmit} className="gap-2"><Zap className="w-4 h-4" />Confirm & Run Analysis</Button>
          </div>
        )}

        {step === "processing" && (
          <Card>
            <CardContent className="py-8 text-center space-y-4">
              <div className="w-10 h-10 border-4 border-border border-t-primary rounded-full animate-spin mx-auto" />
              <p className="font-semibold">Processing reviews…</p>

              {/* {total > 0 && ( */}
              <>
                <div className="w-56 bg-muted rounded-full h-2 overflow-hidden m-auto">
                  <div className="h-full bg-primary rounded-full transition-all duration-500 origin-left" style={{ width: `${(processed / total) * 100}%` }} />
                </div>
                <p className="text-xs font-number text-muted-foreground mt-1">{processed} / {total}</p>
              </>
              {/* )} */}
            </CardContent>
          </Card>
        )}

        {step === "done" && (
          <Card className="border-green-500/30 bg-green-500/5">
            <CardContent className="py-8 text-center space-y-3">
              <CheckCircle2 className="w-12 h-12 text-green-500 mx-auto" />
              <p className="font-semibold text-lg">Analysis Complete</p>
              <p className="text-sm text-muted-foreground">{total} reviews processed and ready for analysis.</p>
              <Button onClick={() => navigate("/")} className="mt-2">View Dashboard →</Button>
            </CardContent>
          </Card>
        )}
      </main>
    </div>
  );
}
