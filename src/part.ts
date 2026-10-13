// part.ts — the persistent decomposition part and its cost model, shared by
// the decomposer and the merge pass.

import type { Mesh } from "./mesh.ts";
import type { Plane } from "./cpu/cut.ts";

export const K_RV = 0.3;

export interface Part {
  mesh: Mesh;
  hull: Mesh;
  meshVol: number;
  hullVol: number;
  hausdorff: number | null;
  /** set when a cut could not be applied; the part is not retried */
  stuck: boolean;
  stuckReason?: "no-plane" | "cut-failed" | "not-closed" | "degenerate";
  /** planes of internal seams left by hull merging (the merged mesh keeps both caps) */
  seams?: Plane[];
  /** number of leaf parts merged into this one (absent or 1 for an unmerged part) */
  merged?: number;
}

export function rvCost(meshVol: number, hullVol: number): number {
  return K_RV * Math.cbrt((3 / (4 * Math.PI)) * Math.max(hullVol - meshVol, 0));
}
export function partCost(p: Part): number {
  return Math.max(rvCost(p.meshVol, p.hullVol), p.hausdorff ?? 0);
}
