import type { Mesh } from "../mesh.js";
export declare const PC_EPS = 0.000001;
export type Plane = [number, number, number, number];
export interface CutResult {
    ok: boolean;
    pos: Mesh | null;
    neg: Mesh | null;
    reason?: string;
}
export interface CapDebug {
    loops: {
        len: number;
        area: number;
        depth: number;
        parent: number;
        tris: number;
    }[];
}
/** Filled by the last planeCut call when `debug` is passed. */
export declare const lastCutDebug: {
    pos: CapDebug | null;
    neg: CapDebug | null;
};
export declare function planeCut(mesh: Mesh, plane: Plane, eps?: number, debug?: boolean): CutResult;
