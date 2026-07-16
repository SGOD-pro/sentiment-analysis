/** Column mapping set by user at upload time. Persisted in localStorage. */
export interface ColumnMap {
  textCol: string;
  catCol?: string;
  dateCol?: string;
  extraCols: string[];
}

const KEY = "sentimetric:columnMap";

export function saveColumnMap(map: ColumnMap) {
  localStorage.setItem(KEY, JSON.stringify(map));
}

export function loadColumnMap(): ColumnMap {
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) return JSON.parse(raw) as ColumnMap;
  } catch { /* ignore */ }
  return { textCol: "text", extraCols: [] };
}
