export interface Mesh {
    verts: Float32Array;
    tris: Uint32Array;
}
export declare function nv(m: Mesh): number;
export declare function nt(m: Mesh): number;
export declare function makeBox(lo: [number, number, number], hi: [number, number, number]): Mesh;
export declare function mergeMeshes(a: Mesh, b: Mesh): Mesh;
/** Unit cube [0,1]^3 (matches tests/conftest.py unit_cube). */
export declare function unitCube(): Mesh;
/** L-shape from two boxes joined along x (matches tests/conftest.py l_shape). */
export declare function lShape(): Mesh;
/** Signed volume via the divergence theorem, anchored at vertex 0. */
export declare function signedVolume(m: Mesh): number;
export declare function volume(m: Mesh): number;
/** True when every directed edge has exactly one reverse twin. */
/**
 * Directed edges in CSR form by source vertex: edge e = (src, dst[e]) of
 * triangle face[e]; edges of vertex v are dst[off[v] .. off[v+1]).
 */
export declare function edgeCSR(m: Mesh): {
    off: Uint32Array;
    dst: Uint32Array;
    face: Uint32Array;
};
/** Closed in the edge-twin sense: every directed edge occurs once and its reverse occurs once. */
export declare function isClosed(m: Mesh): boolean;
export declare function bbox(m: Mesh): {
    lo: [number, number, number];
    hi: [number, number, number];
};
export interface Normalization {
    scale: number;
    center: [number, number, number];
}
/** Center the mesh and scale its largest extent to 2 (CLI convention). */
export declare function normalize(m: Mesh): {
    mesh: Mesh;
    norm: Normalization;
};
export declare function denormalize(m: Mesh, norm: Normalization): Mesh;
/** Parse a binary or ASCII STL. Vertices are welded by exact position. */
export declare function parseSTL(buf: ArrayBuffer): Mesh;
/** Weld a triangle soup (9 floats per triangle) into an indexed mesh. */
export declare function weld(pos: number[] | Float32Array): Mesh;
/** Parse a simple OBJ (v / f lines, polygons fan-triangulated). */
export declare function parseOBJ(text: string): Mesh;
export declare function toOBJ(parts: Mesh[]): string;
