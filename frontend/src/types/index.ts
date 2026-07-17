/** Standard API response envelope from the backend. */
export interface ApiResponse<T = unknown> {
  success: boolean;
  data?: T;
  error_code?: string;
  message?: string;
}

export interface BatchStatus {
  batch_id: string;
  status: "pending" | "processing" | "done" | "failed";
  total_reviews: number;
  processed_count: number;
  filename: string;
  uploaded_at: string;
  processing_duration_seconds?: number;
  batch_size?: number;
}

export interface TrendWeek {
  week: string;
  positive: number;
  neutral: number;
  negative: number;
}

export interface CategorySummary {
  category: string;
  positive: number;
  neutral: number;
  negative: number;
  total: number;
  sentiment_score: number;
}

export interface IssueCount {
  issue_tag: string;
  count: number;
}

/** A review row — known fields + any extra CSV columns as string values. */
export interface Review {
  review_id: string;
  batch_id: string;
  text: string;
  category?: string;
  review_date?: string;
  processed_at: string;
  sentiment: "positive" | "neutral" | "negative";
  confidence_margin: string;
  prob_negative: string;
  prob_neutral: string;
  prob_positive: string;
  issue_tag?: string;
  issue_distance?: string;
  cluster_source?: string;
  [key: string]: unknown; // extra CSV columns
}

export interface ReviewsPage {
  reviews: Review[];
  total: number;
  page: number;
  limit: number;
  total_pages: number;
}

export interface ReviewFilters {
  sentiment?: string;
  category?: string;
  issue_tag?: string;
  from?: string;
  to?: string;
  page?: number;
  limit?: number;
}
