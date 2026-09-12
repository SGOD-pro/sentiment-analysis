/**
 * Session store — powered by Zustand.
 *
 * Session identity (batchId, status, filename, etc.) is safely persisted to localStorage
 * via partialize so large data is NEVER written to localStorage (preventing QuotaExceeded errors).
 *
 * Analytics data (categories, issues, trends, status), reviews, and optimistic corrections
 * are cached in Zustand in-memory state so navigating between Dashboard, Reviews, and Reports
 * avoids refetching from the backend every time.
 */
import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import type {
  BatchStatus,
  CategorySummary,
  Correction,
  IssueCount,
  Review,
  ReviewFilters,
  TrendWeek,
} from "@/types";
import {
  getCategoriesSummary,
  getIssuesDistribution,
  getTrends,
  getBatchStatus,
  getReviews,
} from "@/api/client";

export interface ColumnMapping {
  textCol: string;
  categoryCol?: string;
  dateCol?: string;
  extraCols: string[];
}

export interface SessionState {
  // Session Identity (persisted)
  batchId: string | null;
  status: "idle" | "processing" | "done" | "failed";
  filename: string;
  totalReviews: number;
  columnMapping: ColumnMapping;
  uploadedAt: string;

  // Cached Analytics Data (in-memory Zustand cache)
  categories: CategorySummary[];
  issues: IssueCount[];
  weeks: TrendWeek[];
  batchStatus: BatchStatus | null;
  analyticsBatchId: string | null;
  analyticsLoading: boolean;

  // Cached Reviews Data (in-memory Zustand cache)
  reviews: Review[];
  reviewsTotal: number;
  reviewsTotalPages: number;
  reviewsPage: number;
  reviewsBatchId: string | null;
  reviewsLoading: boolean;
  reviewsFilters: ReviewFilters | null;

  // Optimistic corrections (in-memory Zustand cache)
  corrections: Record<string, Correction>;

  // Session actions
  setSession: (data: {
    batchId: string;
    filename: string;
    totalReviews: number;
    columnMapping: ColumnMapping;
    uploadedAt: string;
  }) => void;
  setStatus: (status: SessionState["status"]) => void;
  setProcessedCount: (count: number) => void;
  clearSession: () => void;

  // Analytics actions
  fetchAnalytics: (batchId: string, force?: boolean) => Promise<void>;
  setCategories: (categories: CategorySummary[]) => void;
  setIssues: (issues: IssueCount[]) => void;
  setWeeks: (weeks: TrendWeek[]) => void;
  setBatchStatus: (batchStatus: BatchStatus | null) => void;

  // Reviews actions
  fetchReviews: (batchId: string, filters?: ReviewFilters, force?: boolean) => Promise<void>;
  setReviews: (reviews: Review[], total: number, totalPages: number, page: number) => void;
  addCorrection: (reviewId: string, correction: Correction) => void;
}

const INITIAL_IDENTITY = {
  batchId: null,
  status: "idle" as const,
  filename: "",
  totalReviews: 0,
  columnMapping: { textCol: "text", extraCols: [] },
  uploadedAt: "",
};

const INITIAL_CACHE = {
  categories: [] as CategorySummary[],
  issues: [] as IssueCount[],
  weeks: [] as TrendWeek[],
  batchStatus: null as BatchStatus | null,
  analyticsBatchId: null as string | null,
  analyticsLoading: false,

  reviews: [] as Review[],
  reviewsTotal: 0,
  reviewsTotalPages: 1,
  reviewsPage: 1,
  reviewsBatchId: null as string | null,
  reviewsLoading: false,
  reviewsFilters: null as ReviewFilters | null,

  corrections: {} as Record<string, Correction>,
};

// Safe storage wrapper to prevent exceptions in private mode or if quota is exceeded
const safeStorage = {
  getItem: (name: string): string | null => {
    try {
      return localStorage.getItem(name);
    } catch {
      return null;
    }
  },
  setItem: (name: string, value: string): void => {
    try {
      localStorage.setItem(name, value);
    } catch (e) {
      console.warn("localStorage write failed:", e);
    }
  },
  removeItem: (name: string): void => {
    try {
      localStorage.removeItem(name);
    } catch {}
  },
};

export const useSessionStore = create<SessionState>()(
  persist(
    (set, get) => ({
      ...INITIAL_IDENTITY,
      ...INITIAL_CACHE,

      setSession: (data) =>
        set({
          ...data,
          status: "processing",
          ...INITIAL_CACHE,
        }),

      setStatus: (status) => set({ status }),

      setProcessedCount: (totalReviews) => set({ totalReviews }),

      clearSession: () => set({ ...INITIAL_IDENTITY, ...INITIAL_CACHE }),

      // Fetch analytics once and cache in Zustand
      fetchAnalytics: async (targetBatchId: string, force = false) => {
        if (!targetBatchId) return;
        const state = get();
        if (
          !force &&
          state.analyticsBatchId === targetBatchId &&
          state.categories.length > 0
        ) {
          return;
        }

        set({ analyticsLoading: true });
        try {
          const [t, c, i, bs] = await Promise.all([
            getTrends(targetBatchId),
            getCategoriesSummary(targetBatchId),
            getIssuesDistribution(targetBatchId),
            getBatchStatus(targetBatchId),
          ]);

          set({
            analyticsBatchId: targetBatchId,
            weeks: t.data?.weeks ?? [],
            categories: c.data?.categories ?? [],
            issues: i.data?.issues ?? [],
            batchStatus: bs.data ?? null,
            analyticsLoading: false,
          });
        } catch (error) {
          console.error("Failed to fetch analytics into Zustand store:", error);
          set({ analyticsLoading: false });
        }
      },

      setCategories: (categories) => set({ categories }),
      setIssues: (issues) => set({ issues }),
      setWeeks: (weeks) => set({ weeks }),
      setBatchStatus: (batchStatus) => set({ batchStatus }),

      // Fetch reviews with filter caching in Zustand
      fetchReviews: async (
        targetBatchId: string,
        filters: ReviewFilters = { page: 1, limit: 25 },
        force = false
      ) => {
        if (!targetBatchId) return;
        const state = get();
        const currentFiltersStr = JSON.stringify(state.reviewsFilters);
        const newFiltersStr = JSON.stringify(filters);

        if (
          !force &&
          state.reviewsBatchId === targetBatchId &&
          state.reviews.length > 0 &&
          currentFiltersStr === newFiltersStr
        ) {
          return;
        }

        set({ reviewsLoading: true });
        try {
          const res = await getReviews(targetBatchId, filters);
          if (res.data) {
            const currentCorrections = get().corrections;
            const reviewsWithCorrections = (res.data.reviews ?? []).map((r) =>
              currentCorrections[r.review_id]
                ? { ...r, correction: currentCorrections[r.review_id] }
                : r
            );

            set({
              reviews: reviewsWithCorrections,
              reviewsTotal: res.data.total ?? 0,
              reviewsTotalPages: res.data.total_pages ?? 1,
              reviewsPage: filters.page ?? 1,
              reviewsBatchId: targetBatchId,
              reviewsFilters: filters,
              reviewsLoading: false,
            });
          } else {
            set({ reviewsLoading: false });
          }
        } catch (error) {
          console.error("Failed to fetch reviews into Zustand store:", error);
          set({ reviewsLoading: false });
        }
      },

      setReviews: (reviews, reviewsTotal, reviewsTotalPages, reviewsPage) =>
        set({ reviews, reviewsTotal, reviewsTotalPages, reviewsPage }),

      addCorrection: (reviewId: string, correction: Correction) => {
        set((state) => ({
          corrections: { ...state.corrections, [reviewId]: correction },
          reviews: state.reviews.map((r) =>
            r.review_id === reviewId ? { ...r, correction } : r
          ),
        }));
      },
    }),
    {
      name: "sentimetric-session",
      storage: createJSONStorage(() => safeStorage),
      // Crucial: Only persist session metadata to localStorage to prevent storage quota issues
      partialize: (state) => ({
        batchId: state.batchId,
        status: state.status,
        filename: state.filename,
        totalReviews: state.totalReviews,
        columnMapping: state.columnMapping,
        uploadedAt: state.uploadedAt,
      }),
    }
  )
);
