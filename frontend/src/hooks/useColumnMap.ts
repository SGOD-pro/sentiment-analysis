/**
 * Column mapping — compatibility shim.
 * Reads/writes through useSessionStore. Existing callers (Dashboard, Reviews)
 * keep working without changes until they migrate to useSessionStore directly.
 */
import { useSessionStore, type ColumnMapping } from "./useSessionStore";

export type ColumnMap = {
  textCol: string;
  catCol?: string;
  dateCol?: string;
  extraCols: string[];
};

export function saveColumnMap(map: ColumnMap) {
  const mapping: ColumnMapping = {
    textCol: map.textCol,
    categoryCol: map.catCol,
    dateCol: map.dateCol,
    extraCols: map.extraCols,
  };
  // Update just the columnMapping in the session store
  useSessionStore.setState({ columnMapping: mapping });
}

export function loadColumnMap(): ColumnMap {
  const { columnMapping } = useSessionStore.getState();
  return {
    textCol: columnMapping.textCol,
    catCol: columnMapping.categoryCol,
    dateCol: columnMapping.dateCol,
    extraCols: columnMapping.extraCols,
  };
}
