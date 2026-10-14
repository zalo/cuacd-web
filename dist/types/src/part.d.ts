import type { Mesh } from "./mesh.js";
import type { Plane } from "./cpu/cut.js";
export declare const K_RV = 0.3;
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
export declare function rvCost(meshVol: number, hullVol: number): number;
export declare function partCost(p: Part): number;
