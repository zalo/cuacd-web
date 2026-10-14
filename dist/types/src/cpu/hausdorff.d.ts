import type { Mesh } from "../mesh.js";
export declare const HD_RESOLUTION = 2000;
export declare const HD_MIN_SAMPLES = 1000;
/** Area-proportional surface samples (xyz interleaved). */
export declare function sampleMesh(m: Mesh): Float32Array;
export declare function distPtTriSq(px: number, py: number, pz: number, ax: number, ay: number, az: number, bx: number, by: number, bz: number, cx: number, cy: number, cz: number): number;
/** Flat binary BVH over triangles (median split along the longest axis). */
export declare class TriBVH {
    readonly V: Float32Array;
    readonly T: Uint32Array;
    readonly bmin: Float32Array;
    readonly bmax: Float32Array;
    readonly left: Int32Array;
    readonly right: Int32Array;
    readonly start: Int32Array;
    readonly count: Int32Array;
    readonly order: Uint32Array;
    nNodes: number;
    constructor(m: Mesh, leafSize?: number);
    private build;
    private boxDistSq;
    /** Squared distance from a point to the nearest triangle. */
    nearestSq(px: number, py: number, pz: number, bestIn?: number): number;
}
/** max over samples of a of the distance to mesh b (one direction). */
export declare function directedDistance(samplesA: Float32Array, bvhB: TriBVH): number;
export declare function hausdorff(a: Mesh, b: Mesh): number;
