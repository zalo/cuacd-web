// part.ts — the persistent decomposition part and its cost model, shared by
// the decomposer and the merge pass.
export const K_RV = 0.3;
export function rvCost(meshVol, hullVol) {
    return K_RV * Math.cbrt((3 / (4 * Math.PI)) * Math.max(hullVol - meshVol, 0));
}
export function partCost(p) {
    return Math.max(rvCost(p.meshVol, p.hullVol), p.hausdorff ?? 0);
}
