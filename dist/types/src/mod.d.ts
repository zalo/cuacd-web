/// <reference types="@webgpu/types" />
export * from "./mesh.js";
export * from "./decompose.js";
export { mergeHulls, seamPlane, filteredConcat, DEFAULT_MERGE_OPTIONS, type MergeOptions, type MergeStats } from "./merge.js";
export { GpuTree, DEFAULT_TREE_CONFIG, type TreeConfig, type TreeRunResult } from "./gpu/tree.js";
export { getDevice } from "./gpu/device.js";
export { planeCut, type Plane, type CutResult } from "./cpu/cut.js";
export { convexHull, initHull, setHullBackend, hullBackendName } from "./cpu/hull.js";
export { hausdorff, sampleMesh, TriBVH } from "./cpu/hausdorff.js";
export { concaveEdgePlanes } from "./cpu/concave.js";
