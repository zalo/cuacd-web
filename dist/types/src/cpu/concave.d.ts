import { type Mesh } from "../mesh.js";
export interface ConcaveOptions {
    nConcaveEdges: number;
    concaveEps: number;
    concaveThreshold: number;
    seed?: number;
}
/** Returns planes as a flat Float32Array (4 floats per plane). */
export declare function concaveEdgePlanes(m: Mesh, o: ConcaveOptions): Float32Array;
