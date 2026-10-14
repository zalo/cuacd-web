import type { Mesh } from "../mesh.js";
/** Hausdorff pairs per call (result slots in the counter buffer) */
export declare const HD_MAX_PAIRS = 1024;
export interface TreeConfig {
    width: number;
    width2: number;
    depth: number;
    nConcaveEdges: number;
    minEdgeDist: number;
    /** pool capacities in elements */
    vertexCap: number;
    triCap: number;
    edgeCap: number;
    scratchWords: number;
    /** byte budget for in-flight hull scratch slabs; the chunk size follows from the slab size */
    hullBudgetBytes: number;
    /** hull vertices a single part may have before its hull is reported as failed */
    hullMaxVerts: number;
    /** branch-and-bound pruning of exact hulls at the last tree level (CuACD la_hull behaviour) */
    pruneLastLevel: boolean;
}
export declare const DEFAULT_TREE_CONFIG: TreeConfig;
export interface ItemView {
    partIds: Uint32Array;
    np: number;
    srcPart: number;
    initialCut: number;
    nLevels: number;
    levelCosts: Float32Array;
    flags: number;
    parentItem: number;
    cutIdx: number;
}
export interface RefineRequest {
    /** probe rounds after the tree (each re-scores offsets of the current plane with the full lookahead) */
    rounds: number;
    /** probes per round (3 .. planeSlots) */
    probes: number;
    /** per-seed initial half-width of the offset scan (0 disables refinement for that seed) */
    spacing: number[];
}
export interface SeedResult {
    bestCut: number;
    bestCost: number;
    plane: [number, number, number, number] | null;
}
export interface TreeRunResult {
    seeds: SeedResult[];
    overflow: boolean;
    hullFail: boolean;
    hullFailCount: number;
    npOverflow: boolean;
    levels: ItemView[][];
    stats: {
        jobsPerLevel: number[];
        gpuMs: number;
        hullSlabWords: number;
        hullChunkSize: number;
        kernelMs?: Record<string, number>;
        pruned?: number;
    };
    /** level-1 child part volumes when `readParts` was requested: 16 floats per part, [6]=meshVol [7]=hullVol */
    partVols?: {
        base: number;
        data: Float32Array;
    };
}
export declare class GpuTree {
    readonly device: GPUDevice;
    readonly cfg: TreeConfig;
    private pipelines;
    private layout;
    private pool;
    private parts;
    private items;
    private jobs;
    private planes;
    private scratch;
    private ctr;
    private params;
    private bindGroup;
    private capacity;
    private readonly edgeStride;
    private ready;
    /** when true and the device supports timestamp queries, run() reports per-kernel GPU time */
    profile: boolean;
    private ctrInit;
    private querySet;
    private queryBuf;
    constructor(device: GPUDevice, cfg?: Partial<TreeConfig>);
    private init;
    /** (Re)allocate the per-run tables for up to `nSeeds` seeds. */
    /** Grow only the part table (for hull / Hausdorff batches that do not expand a tree). */
    private ensurePartSlots;
    private ensureTables;
    private rebuildBindGroup;
    private uploadSeeds;
    private paramsWords;
    private hullSlab;
    /**
     * Bidirectional sampled Hausdorff distance for mesh pairs (part, hull) on
     * the GPU. Pairs are uploaded as seeds: A_k at index k, B_k at n + k.
     */
    hausdorff(pairs: {
        a: Mesh;
        b: Mesh;
    }[]): Promise<Float32Array>;
    /**
     * Convex hulls of whole meshes on the GPU (same kernel as the tree search).
     * Returns outward-wound hull triangles over each mesh's own vertex indices,
     * or null for a mesh whose hull failed (caller falls back to the CPU).
     */
    /** GPU time of the last profiled hullMeshes call (ms), when `profile` is on */
    lastHullGpuMs: number;
    hullMeshes(meshes: Mesh[], opts?: {
        hullStage?: number;
        noExport?: boolean;
    }): Promise<({
        tris: Uint32Array;
        volume: number;
    } | null)[]>;
    /**
     * Run the tree for the given seeds. `edgePlanes[k]` holds up to
     * 4*nConcaveEdges planes (a,b,c,d) for seed k; zero planes are skipped.
     */
    run(seeds: Mesh[], edgePlanes?: Float32Array[], opts?: {
        depth?: number;
        width?: number;
        nEdgeCuts?: number;
        readParts?: boolean;
        refine?: RefineRequest;
    }): Promise<TreeRunResult>;
    /**
     * Convex hulls and Hausdorff(part, hull) distances of whole meshes in one
     * submission: the hull kernel exports the faces and a hull descriptor, the
     * Hausdorff kernel samples both in the same command buffer.
     */
    /** failure codes of the last hullsWithHausdorff call (0 = ok, 1 = visible set overflow, 2 = face slots exhausted, 3 = slab too small) */
    lastHullFails: number[];
    hullsWithHausdorff(meshes: Mesh[]): Promise<({
        tris: Uint32Array;
        volume: number;
        hausdorff: number;
    } | null)[]>;
    /** Planes per seed available for edge/probe cuts. */
    get planeSlots(): number;
    /** Debug: read a part's descriptor and geometry back (uncapped mesh). */
    readPart(partIdx: number): Promise<{
        mesh: Mesh;
        edges: Float32Array;
        meshVol: number;
        hullVol: number;
        nPlanes: number;
        planeIds: number[];
    }>;
    /** Debug: alive hull faces of child `childIdx` (index within level 0) from the last run's slab. */
    readHullFaces(childIdx: number): Promise<{
        faces: Uint32Array;
        nSlots: number;
    }>;
    private lastHullSlabWords;
    private lastHullChunkSize;
    private lastPartBase;
    readPlane(planeIdx: number): Promise<[number, number, number, number]>;
    get visMax(): number;
    destroy(): void;
}
