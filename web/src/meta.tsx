import { createContext, useContext } from "react";
import type { Meta } from "./api";

export const MetaContext = createContext<Meta | null>(null);
export function useMeta(): Meta {
  const m = useContext(MetaContext);
  if (!m) throw new Error("MetaContext missing");
  return m;
}
