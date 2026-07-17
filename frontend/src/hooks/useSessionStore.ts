/**
 * Session store — persisted via localStorage.
 *
 * Holds session IDENTITY only (batchId, status, filename, column mapping).
 * Do NOT store dashboard payloads here — refetch them using batchId.
 */
import { create } from "zustand";
import { persist } from "zustand/middleware";

export interface ColumnMapping {
  textCol: string;
  categoryCol?: string;
  dateCol?: string;
  extraCols: string[];
}

interface SessionState {
  batchId: string | null;
  status: "idle" | "processing" | "done" | "failed";
  filename: string;
  totalReviews: number;
  columnMapping: ColumnMapping;
  uploadedAt: string;

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
}

const INITIAL: Pick<SessionState, "batchId" | "status" | "filename" | "totalReviews" | "columnMapping" | "uploadedAt"> = {
  batchId: null,
  status: "idle",
  filename: "",
  totalReviews: 0,
  columnMapping: { textCol: "text", extraCols: [] },
  uploadedAt: "",
};

export const useSessionStore = create<SessionState>()(
  persist(
    (set) => ({
      ...INITIAL,
      setSession: (data) => set({ ...data, status: "processing" }),
      setStatus: (status) => set({ status }),
      setProcessedCount: (totalReviews) => set({ totalReviews }),
      clearSession: () => set(INITIAL),
    }),
    { name: "sentimetric-session" },
  ),
);
