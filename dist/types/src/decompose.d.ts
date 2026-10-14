import { type TreeConfig } from "./gpu/tree.js";
import { type Plane } from "./cpu/cut.js";
import { type Mesh, type Normalization } from "./mesh.js";
export { K_RV, type Part, partCost, rvCost } from "./part.js";
import { type Part } from "./part.js";
import { type MergeStats } from "./merge.js";
export interface DecomposeOptions {
    threshold: number;
    maxIters: number;
    width: number;
    width2: number;
    depth: number;
    maxNCutting: number;
    nConcaveEdges: number;
    concaveEps: number;
    concaveThreshold: number;
    concaveIters: number;
    /** center and scale the input so the largest extent is 2 (CLI behaviour) */
    normalize: boolean;
    /** reject input that is not a closed manifold (every directed edge twinned once) */
    requireClosed: boolean;
    /** verify that every CPU cut result is closed; failed cuts leave the part untouched */
    validateCuts: boolean;
    /** hull implementation for the persistent parts; the tree search always uses the GPU */
    hullBackend: "gpu" | "cpu";
    /** Hausdorff implementation for the stopping criterion */
    hausdorffBackend: "gpu" | "cpu";
    /** refinement rounds that scan offsets along the chosen plane's normal (0 = off) */
    refineRounds: number;
    /** probes per refinement round (<= 32) */
    refineProbes: number;
    /** greedy hull-merging post-pass (CoACD MergeConvexHulls; the CuACD CLI default) */
    mergeHulls: boolean;
    /** maximum merge rounds; each round re-evaluates only pairs that involve a new part */
    mergeRounds: number;
    /** vertex-to-plane tolerance for ignoring seam triangles in the merge Hausdorff */
    mergePlaneTol: number;
    verbose: number;
    /** per-kernel GPU timing (needs the timestamp-query feature) */
    profile: boolean;
    tree: Partial<TreeConfig>;
    /** aborts at the next iteration or merge-round boundary with an AbortError */
    signal?: AbortSignal;
    onProgress?: (p: ProgressInfo) => void;
    /** debugging hook: called when a winning cut cannot be applied */
    onCutFailure?: (mesh: Mesh, plane: Plane, reason: string) => void;
}
export type ProgressPhase = "start" | "select" | "tree" | "cut" | "hull" | "iteration" | "merge" | "done";
export interface ProgressInfo {
    /** what is about to run (or, for "iteration" and "done", what just finished) */
    phase: ProgressPhase;
    iteration: number;
    nParts: number;
    nCutting: number;
    worstCost: number;
    /** monotone overall progress estimate in [0, 1] (from the worst part cost and the iteration count) */
    fraction: number;
    message: string;
    gpuMs: number;
    cpuMs: number;
}
export declare const DEFAULT_OPTIONS: DecomposeOptions;
export interface DecomposeResult {
    parts: Part[];
    iterations: number;
    stats: {
        gpuMs: number;
        cpuMs: number;
        totalMs: number;
        treeRuns: number;
        hullFailRuns: number;
        hausdorffMs: number;
        cutMs: number;
        concaveMs: number;
        hullMs: number;
        treeMs: number;
        mergeMs: number;
        kernelMs: Record<string, number>;
    };
    /** merge-pass statistics (null when mergeHulls is off or there was nothing to merge) */
    merge: MergeStats | null;
    norm: Normalization | null;
}
export declare class Decomposer {
    readonly device: GPUDevice;
    readonly opts: DecomposeOptions;
    private tree;
    /** epsilon for cleaning output hulls: above the float32 resolution of the input coordinates (set per decompose) */
    private hullEps;
    constructor(device: GPUDevice, opts?: Partial<DecomposeOptions>);
    destroy(): void;
    /** Hausdorff(part, hull) for parts that lack it, batched on the GPU (CPU fallback). */
    private fillHausdorff;
    /** Parts (mesh + hull + volumes) for a batch of meshes; GPU hulls with CPU fallback. */
    private makeParts;
    /** Initial half-width of the refinement scan per part: the level-0 grid spacing of its smallest axis. */
    private refineSpacing;
    decompose(input: Mesh): Promise<DecomposeResult>;
}
/** Convenience: acquire a device, decompose, release. */
export declare function decompose(mesh: Mesh, opts?: Partial<DecomposeOptions>): Promise<DecomposeResult>;
