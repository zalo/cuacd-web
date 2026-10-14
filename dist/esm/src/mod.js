/// <reference types="@webgpu/types" />
// cuacd-web — WebGPU/CPU hybrid port of CuACD's lookahead convex decomposition.
import "../_dnt.polyfills.js";
export * from "./mesh.js";
export * from "./decompose.js";
export { mergeHulls, seamPlane, filteredConcat, DEFAULT_MERGE_OPTIONS } from "./merge.js";
export { GpuTree, DEFAULT_TREE_CONFIG } from "./gpu/tree.js";
export { getDevice } from "./gpu/device.js";
export { planeCut } from "./cpu/cut.js";
export { convexHull, initHull, setHullBackend, hullBackendName } from "./cpu/hull.js";
export { hausdorff, sampleMesh, TriBVH } from "./cpu/hausdorff.js";
export { concaveEdgePlanes } from "./cpu/concave.js";
