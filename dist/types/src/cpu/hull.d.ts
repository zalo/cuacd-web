import type { Mesh } from "../mesh.js";
export interface HullResult {
    mesh: Mesh;
    volume: number;
}
export interface HullBackend {
    name: string;
    /** eps is relative to the largest coordinate magnitude (Kuukka's convention); default 1e-7 */
    compute(points: Float64Array, n: number, eps?: number): {
        verts: Float64Array;
        tris: Uint32Array;
    };
}
export declare function setHullBackend(b: HullBackend | null): void;
export declare function hullBackendName(): string;
/** Load the Emscripten quickhull (wasm/qh.js) as the active backend (idempotent). */
export declare function initHull(): Promise<boolean>;
/** Convex hull of the vertices of a mesh or point array; triangles wound outward. */
export declare function convexHull(verts: Float32Array | Float64Array): HullResult;
/** Build an outward-wound hull mesh from triangles indexing `hv`, dropping unused vertices. */
/**
 * Clean the faces of an exact (grid-quantized) hull for output. The grid
 * makes points that are collinear or coplanar in float coordinates into hull
 * vertices, so the raw export carries needle faces and extra vertices. The
 * hull is rebuilt in double precision from the raw hull's own vertex set (a
 * few hundred well-spread points, which the CPU quickhull handles reliably)
 * and accepted only when it is closed, keeps the volume and contains every
 * raw vertex; otherwise the raw faces are returned. `eps` (relative to the
 * extent) must stay well above the float32 resolution of the coordinates the
 * hull will be stored in: faces thinner than that get unreliable normals.
 */
export declare function cleanHullFaces(V: Float32Array | Float64Array, T: Uint32Array, eps?: number): Uint32Array;
export declare function compactHull(hv: Float32Array | Float64Array, tris: Uint32Array): HullResult;
