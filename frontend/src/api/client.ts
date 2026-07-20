import type {
  ApiResponse,
  BatchStatus,
  CategorySummary,
  IssueCount,
  Review,
  ReviewFilters,
  ReviewsPage,
  TrendWeek,
} from "../types";

const BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<ApiResponse<T>> {
  const res = await fetch(`${BASE}${path}`, init);
  return res.json();
}

export async function uploadCSV(
  file: File,
  textCol: string,
  categoryCol?: string,
  dateCol?: string,
): Promise<ApiResponse<{ batch_id: string }>> {
  const form = new FormData();
  form.append("file", file);
  form.append("text_col", textCol);
  if (categoryCol) form.append("category_col", categoryCol);
  if (dateCol) form.append("date_col", dateCol);
  return request("/api/upload", { method: "POST", body: form });
}

export function getBatchStatus(batchId: string) {
  return request<BatchStatus>(`/api/batches/${batchId}/status`);
}

export function getTrends(batchId: string, from?: string, to?: string, category?: string) {
  const params = new URLSearchParams();
  params.set("batch_id", batchId);
  if (from) params.set("from", from);
  if (to) params.set("to", to);
  if (category) params.set("category", category);
  return request<{ weeks: TrendWeek[] }>(`/api/trends?${params}`);
}

export function getCategoriesSummary(batchId: string, from?: string, to?: string) {
  const params = new URLSearchParams();
  params.set("batch_id", batchId);
  if (from) params.set("from", from);
  if (to) params.set("to", to);
  return request<{ categories: CategorySummary[] }>(`/api/categories/summary?${params}`);
}

export function getIssuesDistribution(batchId: string, from?: string, to?: string, category?: string) {
  const params = new URLSearchParams();
  params.set("batch_id", batchId);
  if (from) params.set("from", from);
  if (to) params.set("to", to);
  if (category) params.set("category", category);
  return request<{ issues: IssueCount[] }>(`/api/issues/distribution?${params}`);
}

export function getReviews(batchId: string, filters: ReviewFilters = {}) {
  const params = new URLSearchParams();
  params.set("batch_id", batchId);
  if (filters.sentiment) params.set("sentiment", filters.sentiment);
  if (filters.category) params.set("category", filters.category);
  if (filters.issue_tag) params.set("issue_tag", filters.issue_tag);
  if (filters.from) params.set("from", filters.from);
  if (filters.to) params.set("to", filters.to);
  if (filters.page) params.set("page", String(filters.page));
  if (filters.limit) params.set("limit", String(filters.limit));
  return request<ReviewsPage>(`/api/reviews?${params}`);
}

export function getReview(reviewId: string, batchId: string) {
  const params = new URLSearchParams();
  params.set("batch_id", batchId);
  return request<Review>(`/api/reviews/${reviewId}?${params}`);
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await request<{ status: string }>("/health");
    return res.success === true && res.data?.status === "healthy";
  } catch {
    return false;
  }
}
