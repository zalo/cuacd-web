// cuacd-web — WebGPU/CPU hybrid port of CuACD's lookahead convex decomposition.
export * from "./mesh.ts";
export * from "./decompose.ts";
export { mergeHulls, seamPlane, filteredConcat, DEFAULT_MERGE_OPTIONS, type MergeOptions, type MergeStats } from "./merge.ts";
export { GpuTree, DEFAULT_TREE_CONFIG, type TreeConfig, type TreeRunResult } from "./gpu/tree.ts";
export { getDevice } from "./gpu/device.ts";
export { planeCut, type Plane, type CutResult } from "./cpu/cut.ts";
export { convexHull, initHull, setHullBackend, hullBackendName } from "./cpu/hull.ts";
export { hausdorff, sampleMesh, TriBVH } from "./cpu/hausdorff.ts";
export { concaveEdgePlanes } from "./cpu/concave.ts";
