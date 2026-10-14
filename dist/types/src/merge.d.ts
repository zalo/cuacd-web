import { type Mesh } from "./mesh.js";
import type { Plane } from "./cpu/cut.js";
import type { GpuTree } from "./gpu/tree.js";
import { type Part } from "./part.js";
export interface MergeOptions {
    threshold: number;
    /** maximum matching rounds (the pass stops earlier when nothing merges) */
    maxRounds: number;
    /** distance within which a vertex counts as lying on a seam plane */
    planeTol: number;
    hullBackend: "gpu" | "cpu";
    hausdorffBackend: "gpu" | "cpu";
    verbose: number;
    /** epsilon for cleaning merged hull meshes (see cleanHullFaces) */
    hullEps: number;
    /** aborts at the next round boundary with an AbortError */
    signal?: AbortSignal;
    /** called at the start of a round (merged = -1) and at its end */
    onRound?: (info: {
        round: number;
        nParts: number;
        candidates: number;
        merged: number;
    }) => void;
    /** debugging hook: every pair that got a merged hull, with its costs */
    onPair?: (info: {
        a: Part;
        b: Part;
        rv: number;
        hd: number;
        cost: number;
        seam: Plane | null;
    }) => void;
}
export declare const DEFAULT_MERGE_OPTIONS: MergeOptions;
export interface MergeStats {
    rounds: number;
    merges: number;
    /** pairs that passed the cheap rejections and got a merged hull */
    pairsHulled: number;
    /** pairs that also needed a Hausdorff distance */
    pairsHausdorff: number;
    /** merged hulls that fell back to the CPU hull */
    hullFallbacks: number;
    hullMs: number;
    hausdorffMs: number;
    totalMs: number;
}
/**
 * Seam plane between two parts: a face plane of A's hull with all of B's hull
 * on its outer side (or on it), choosing the one with the largest coplanar
 * mesh area in A and B. Null when no separating face carries any mesh area.
 * Planes use the cut.ts convention (a, b, c, d) with a x + b y + c z + d = 0.
 */
export declare function seamPlane(a: Part, b: Part, tol: number): Plane | null;
/** Concatenation of two meshes without the triangles lying on any of the given planes. */
export declare function filteredConcat(a: Mesh, b: Mesh, planes: Plane[], tol: number): Mesh;
/**
 * Merge parts whose union is convex enough (cost below the threshold).
 * Returns the new part list; the input array is not modified.
 */
export declare function mergeHulls(tree: GpuTree | null, input: Part[], o?: Partial<MergeOptions>): Promise<{
    parts: Part[];
    stats: MergeStats;
}>;
