// decompose.ts — lookahead approximate convex decomposition (port of
// csrc/lookahead.c + la_*.cu to a WebGPU/CPU hybrid).
//
// Per iteration:
//   1. Hausdorff(part, hull) for parts that do not have it yet (CPU)
//   2. pick the highest-cost parts above threshold (up to maxNCutting)
//   3. concave-edge candidate planes for those parts (CPU)
//   4. GPU tree search over axis + edge planes → best first cut per part
//   5. apply the winning plane with a capped CPU cut; hulls on the CPU
//
// The tree search ranks cuts by the rv (volume-ratio) cost only while the
// stopping criterion uses max(rv, hausdorff), exactly as CoACD/CuACD do.

import { GpuTree, type TreeConfig } from "./gpu/tree.ts";
import { getDevice } from "./gpu/device.ts";
import { planeCut, type Plane } from "./cpu/cut.ts";
import { cleanHullFaces, compactHull, convexHull, initHull } from "./cpu/hull.ts";
import { hausdorff } from "./cpu/hausdorff.ts";
import { concaveEdgePlanes } from "./cpu/concave.ts";
import { bbox, denormalize, isClosed, type Mesh, normalize, type Normalization, volume } from "./mesh.ts";

export { K_RV, type Part, partCost, rvCost } from "./part.ts";
import { type Part, partCost } from "./part.ts";
import { mergeHulls, type MergeStats } from "./merge.ts";

export interface DecomposeOptions {
  threshold: number;
  maxIters: number;
  width: number;
  width2: number;
  depth: number;
  maxNCutting: number;
  nConcaveEdges: number;
  concaveEps: number;
  concaveThreshold: number;
  concaveIters: number;
  /** center and scale the input so the largest extent is 2 (CLI behaviour) */
  normalize: boolean;
  /** reject input that is not a closed manifold (every directed edge twinned once) */
  requireClosed: boolean;
  /** verify that every CPU cut result is closed; failed cuts leave the part untouched */
  validateCuts: boolean;
  /** hull implementation for the persistent parts; the tree search always uses the GPU */
  hullBackend: "gpu" | "cpu";
  /** Hausdorff implementation for the stopping criterion */
  hausdorffBackend: "gpu" | "cpu";
  /** refinement rounds that scan offsets along the chosen plane's normal (0 = off) */
  refineRounds: number;
  /** probes per refinement round (<= 32) */
  refineProbes: number;
  /** greedy hull-merging post-pass (CoACD MergeConvexHulls; the CuACD CLI default) */
  mergeHulls: boolean;
  /** maximum merge rounds; each round re-evaluates only pairs that involve a new part */
  mergeRounds: number;
  /** vertex-to-plane tolerance for ignoring seam triangles in the merge Hausdorff */
  mergePlaneTol: number;
  verbose: number;
  /** per-kernel GPU timing (needs the timestamp-query feature) */
  profile: boolean;
  tree: Partial<TreeConfig>;
  /** aborts at the next iteration or merge-round boundary with an AbortError */
  signal?: AbortSignal;
  onProgress?: (p: ProgressInfo) => void;
  /** debugging hook: called when a winning cut cannot be applied */
  onCutFailure?: (mesh: Mesh, plane: Plane, reason: string) => void;
}

export type ProgressPhase = "start" | "select" | "tree" | "cut" | "hull" | "iteration" | "merge" | "done";

export interface ProgressInfo {
  /** what is about to run (or, for "iteration" and "done", what just finished) */
  phase: ProgressPhase;
  iteration: number;
  nParts: number;
  nCutting: number;
  worstCost: number;
  /** monotone overall progress estimate in [0, 1] (from the worst part cost and the iteration count) */
  fraction: number;
  message: string;
  gpuMs: number;
  cpuMs: number;
}

export const DEFAULT_OPTIONS: DecomposeOptions = {
  threshold: 0.05,
  maxIters: 100,
  width: 30,
  width2: 3, // CuACD's width2=5 yields 3 valid axis cuts (cpa = 5/3)
  depth: 2,
  maxNCutting: 16,
  nConcaveEdges: 16,
  concaveEps: 0.005,
  concaveThreshold: 3.49,
  concaveIters: 10,
  normalize: true,
  requireClosed: true,
  validateCuts: true,
  hullBackend: "gpu",
  hausdorffBackend: "gpu",
  refineRounds: 2,
  refineProbes: 8,
  mergeHulls: true,
  mergeRounds: 32,
  mergePlaneTol: 1e-3,
  verbose: 0,
  profile: false,
  tree: {},
};

export interface DecomposeResult {
  parts: Part[];
  iterations: number;
  stats: {
    gpuMs: number; cpuMs: number; totalMs: number; treeRuns: number; hullFailRuns: number;
    hausdorffMs: number; cutMs: number; concaveMs: number; hullMs: number; treeMs: number; mergeMs: number;
    kernelMs: Record<string, number>;
  };
  /** merge-pass statistics (null when mergeHulls is off or there was nothing to merge) */
  merge: MergeStats | null;
  norm: Normalization | null;
}

function makePartCpu(mesh: Mesh): Part {
  const h = convexHull(mesh.verts);
  return { mesh, hull: h.mesh, meshVol: volume(mesh), hullVol: h.volume, hausdorff: null, stuck: false };
}

export class Decomposer {
  readonly device: GPUDevice;
  readonly opts: DecomposeOptions;
  private tree: GpuTree;
  /** epsilon for cleaning output hulls: above the float32 resolution of the input coordinates (set per decompose) */
  private hullEps = 2e-5;

  constructor(device: GPUDevice, opts: Partial<DecomposeOptions> = {}) {
    this.device = device;
    this.opts = { ...DEFAULT_OPTIONS, ...opts, tree: { ...(opts.tree ?? {}) } };
    const o = this.opts;
    // width must be a multiple of 3 (cuts per axis)
    if (o.width % 3 !== 0 || o.width2 % 3 !== 0) throw new Error("width and width2 must be multiples of 3");
    this.tree = new GpuTree(device, {
      width: o.width,
      width2: o.width2,
      depth: o.depth,
      nConcaveEdges: o.nConcaveEdges,
      minEdgeDist: Math.min(o.threshold / 4, 0.015),
      ...o.tree,
    });
    this.tree.profile = o.profile;
  }

  destroy() {
    this.tree.destroy();
  }

  /** Hausdorff(part, hull) for parts that lack it, batched on the GPU (CPU fallback). */
  private async fillHausdorff(parts: Part[]) {
    const todo = parts.filter((p) => p.hausdorff === null);
    if (todo.length === 0) return;
    if (this.opts.hausdorffBackend === "cpu") {
      for (const p of todo) p.hausdorff = hausdorff(p.mesh, p.hull);
      return;
    }
    const batch = 64;
    for (let i = 0; i < todo.length; i += batch) {
      const chunk = todo.slice(i, i + batch);
      const hd = await this.tree.hausdorff(chunk.map((p) => ({ a: p.mesh, b: p.hull })));
      chunk.forEach((p, k) => { p.hausdorff = Number.isFinite(hd[k]) ? hd[k] : hausdorff(p.mesh, p.hull); });
    }
  }

  /** Parts (mesh + hull + volumes) for a batch of meshes; GPU hulls with CPU fallback. */
  private async makeParts(meshes: Mesh[]): Promise<Part[]> {
    if (meshes.length === 0) return [];
    if (this.opts.hullBackend === "cpu") return meshes.map(makePartCpu);
    const fused = this.opts.hausdorffBackend === "gpu";
    const out: Part[] = [];
    const batch = 256;
    for (let s = 0; s < meshes.length; s += batch) {
      const chunk = meshes.slice(s, s + batch);
      const hulls = fused ? await this.tree.hullsWithHausdorff(chunk) : await this.tree.hullMeshes(chunk);
      chunk.forEach((mesh, i) => {
        const h = hulls[i];
        if (!h) {
          if (this.opts.verbose) console.log(`[hull] GPU hull failed for a part with ${mesh.verts.length / 3} vertices (code ${this.tree.lastHullFails[i] ?? "?"}); CPU fallback`);
          out.push(makePartCpu(mesh));
          return;
        }
        const hm = compactHull(mesh.verts, cleanHullFaces(mesh.verts, h.tris, this.hullEps));
        const hd = fused ? (h as unknown as { hausdorff: number }).hausdorff : NaN;
        // the kernel's volume is exact on its grid; the cleaned mesh is for output
        out.push({ mesh, hull: hm.mesh, meshVol: volume(mesh), hullVol: h.volume, hausdorff: Number.isFinite(hd) ? hd : null, stuck: false });
      });
    }
    return out;
  }

  /** Initial half-width of the refinement scan per part: the level-0 grid spacing of its smallest axis. */
  private refineSpacing(meshes: Mesh[]): number[] {
    const o = this.opts;
    const minEdge = Math.min(o.threshold / 4, 0.015);
    const cpa = Math.max(1, Math.floor(o.width / 3));
    return meshes.map((m) => {
      const { lo, hi } = bbox(m);
      let s = Infinity;
      for (let a = 0; a < 3; a++) {
        const valid = hi[a] - lo[a] - 2 * minEdge;
        if (valid > 0) s = Math.min(s, valid / (cpa + 1));
      }
      return Number.isFinite(s) ? s : 0;
    });
  }

  async decompose(input: Mesh): Promise<DecomposeResult> {
    const o = this.opts;
    const tStart = performance.now();
    if (!(await initHull())) throw new Error("failed to load the quickhull wasm module");
    if (o.requireClosed && !isClosed(input)) {
      throw new Error("input mesh is not closed (open or non-manifold edges); repair or remesh it first, or set requireClosed: false");
    }
    let norm: Normalization | null = null;
    let mesh = input;
    if (o.normalize) {
      const r = normalize(input);
      mesh = r.mesh;
      norm = r.norm;
    }
    // Output hulls are stored in the input's float32 frame; a model far from
    // the origin has coarse absolute resolution there, so faces thinner than
    // a few float32 ulps of the largest input coordinate must not survive.
    {
      const { lo, hi } = bbox(input);
      const maxAbs = Math.max(...lo.map(Math.abs), ...hi.map(Math.abs));
      const extent = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) || 1;
      this.hullEps = Math.max(2e-5, 64 * 6e-8 * (maxAbs / extent));
    }
    const checkAbort = () => {
      if (o.signal?.aborted) throw new DOMException("decomposition aborted", "AbortError");
    };
    // progress: a monotone estimate from the worst part cost (log scale from
    // the first iteration's worst cost down to the threshold) and the iteration
    // count, with fixed sub-steps inside an iteration
    let worst0 = 0, lastFrac = 0;
    const estimate = (worst: number, iter: number) => {
      let f = iter / Math.max(1, o.maxIters);
      if (worst0 > o.threshold && worst > 0) f = Math.max(f, Math.log(worst0 / Math.max(worst, o.threshold)) / Math.log(worst0 / o.threshold));
      return Math.min(1, Math.max(0, f));
    };
    const report = (phase: ProgressPhase, iteration: number, nParts: number, nCutting: number, worst: number, frac: number, message: string) => {
      lastFrac = Math.max(lastFrac, Math.min(1, frac));
      o.onProgress?.({ phase, iteration, nParts, nCutting, worstCost: worst, fraction: lastFrac, message, gpuMs, cpuMs });
    };
    let gpuMs = 0, cpuMs = 0, treeRuns = 0, hullFailRuns = 0;
    checkAbort();
    report("start", -1, 0, 0, 0, 0.01, "hull of the input");
    let parts: Part[] = await this.makeParts([mesh]);
    let hausdorffMs = 0, cutMs = 0, concaveMs = 0, hullMs = 0, treeMs = 0;
    const kernelMs: Record<string, number> = {};
    const addKernels = (k?: Record<string, number>) => { if (k) for (const [n, v] of Object.entries(k)) kernelMs[n] = (kernelMs[n] ?? 0) + v; };
    let iterations = 0;
    for (let iter = 0; iter < o.maxIters; iter++) {
      iterations = iter + 1;
      checkAbort();
      const tc0 = performance.now();
      await this.fillHausdorff(parts);
      hausdorffMs += performance.now() - tc0;
      // candidates: ascending by full cost; take the last maxNCutting above threshold
      const order = parts.map((p, i) => ({ i, c: partCost(p) })).sort((a, b) => a.c - b.c);
      const above = order.filter((e) => e.c >= o.threshold && !parts[e.i].stuck);
      const cutting = above.slice(Math.max(0, above.length - o.maxNCutting)).map((e) => e.i);
      const worst = order.length ? order[order.length - 1].c : 0;
      if (iter === 0) worst0 = worst;
      // this iteration spans from the current estimate part of the way to the end
      const est = estimate(worst, iter);
      const span = 0.87 * Math.max(0.02, (1 - est) / 3);
      const base = Math.max(lastFrac, 0.03 + 0.87 * est);
      if (cutting.length === 0) {
        cpuMs += performance.now() - tc0;
        report("select", iter, parts.length, 0, worst, base, `iteration ${iter}: ${parts.length} parts, every part is under the threshold`);
        break;
      }
      report("select", iter, parts.length, cutting.length, worst, base, `iteration ${iter}: ${parts.length} parts, worst cost ${worst.toFixed(4)}, cutting ${cutting.length}`);
      const tcc = performance.now();
      const edgePlanes: Float32Array[] = cutting.map((i) =>
        iter < o.concaveIters && o.nConcaveEdges > 0
          ? concaveEdgePlanes(parts[i].mesh, { nConcaveEdges: o.nConcaveEdges, concaveEps: o.concaveEps, concaveThreshold: o.concaveThreshold, seed: iter * 7919 + i })
          : new Float32Array(0)
      );
      concaveMs += performance.now() - tcc;
      cpuMs += performance.now() - tc0;
      checkAbort();
      report("tree", iter, parts.length, cutting.length, worst, base + 0.1 * span, `iteration ${iter}: lookahead search over ${cutting.length} parts`);

      // GPU tree search, halving the batch on pool overflow
      const planesOut: (Plane | null)[] = new Array(cutting.length).fill(null);
      const costsOut: number[] = new Array(cutting.length).fill(Infinity);
      let batch = cutting.length;
      let start = 0;
      while (start < cutting.length) {
        const idx = cutting.slice(start, start + batch);
        const tt0 = performance.now();
        const seedMeshes = idx.map((i) => parts[i].mesh);
        // the tree and the plane refinement (CoACD's ternary search as batched
        // offset scans scored with the full lookahead) run in one submission
        const nProbes = Math.min(o.refineProbes, this.tree.planeSlots);
        const refine = o.refineRounds > 0 && nProbes >= 3 ? { rounds: o.refineRounds, probes: nProbes, spacing: this.refineSpacing(seedMeshes) } : undefined;
        const res = await this.tree.run(seedMeshes, edgePlanes.slice(start, start + batch), { refine });
        treeMs += performance.now() - tt0;
        treeRuns++;
        gpuMs += res.stats.gpuMs;
        addKernels(res.stats.kernelMs);
        if (o.verbose >= 2) {
          const k = res.stats.kernelMs ?? {};
          console.log(`[iter ${iter}] tree run: seeds nv=[${idx.map((i) => parts[i].mesh.verts.length / 3).join(",")}] jobs=${res.stats.jobsPerLevel.join("/")} slab=${res.stats.hullSlabWords} chunk=${res.stats.hullChunkSize} hull0=${(k.hull0 ?? 0).toFixed(1)} hull1=${(k.hull1 ?? 0).toFixed(1)} emit0=${(k.emit0 ?? 0).toFixed(1)} wall=${(performance.now() - tt0).toFixed(0)}`);
        }
        if (res.overflow) {
          if (batch === 1) throw new Error("GPU pool overflow with a single part; increase tree pool sizes");
          batch = Math.max(1, batch >> 1);
          continue;
        }
        if (res.npOverflow) throw new Error("tree depth exceeds LA_MAX_PARTS");
        if (res.hullFail) {
          hullFailRuns++;
          if (o.verbose) console.log(`[iter ${iter}] warning: some GPU hulls failed (slab ${res.stats.hullSlabWords} words, chunk ${res.stats.hullChunkSize})`);
        }
        res.seeds.forEach((s, k) => { planesOut[start + k] = s.plane; costsOut[start + k] = s.bestCost; });
        start += idx.length;
      }

      // apply the winning cuts on the CPU, then hull all new halves in one GPU batch
      checkAbort();
      report("cut", iter, parts.length, cutting.length, worst, base + 0.6 * span, `iteration ${iter}: cutting ${cutting.length} parts`);
      const ta0 = performance.now();
      const halves: Mesh[] = [];
      const replaced: number[] = [];
      cutting.forEach((pi, k) => {
        const plane = planesOut[k];
        const p = parts[pi];
        if (!plane) { p.stuck = true; p.stuckReason = "no-plane"; return; }
        const r = planeCut(p.mesh, plane);
        if (!r.ok || !r.pos || !r.neg) { p.stuck = true; p.stuckReason = "cut-failed"; o.onCutFailure?.(p.mesh, plane, r.reason ?? "cut-failed"); return; }
        if (r.pos.verts.length < 12 || r.neg.verts.length < 12) { p.stuck = true; p.stuckReason = "degenerate"; o.onCutFailure?.(p.mesh, plane, "degenerate"); return; }
        if (o.validateCuts && (!isClosed(r.pos) || !isClosed(r.neg))) { p.stuck = true; p.stuckReason = "not-closed"; o.onCutFailure?.(p.mesh, plane, "not-closed"); return; }
        replaced.push(pi);
        halves.push(r.pos, r.neg);
      });
      cutMs += performance.now() - ta0;
      cpuMs += performance.now() - ta0;
      checkAbort();
      report("hull", iter, parts.length, cutting.length, worst, base + 0.8 * span, `iteration ${iter}: hulls and Hausdorff distances of ${halves.length} halves`);
      const th0 = performance.now();
      const made = await this.makeParts(halves);
      hullMs += performance.now() - th0;
      gpuMs += performance.now() - th0;
      replaced.forEach((pi, k) => {
        parts[pi] = made[2 * k];
        parts.push(made[2 * k + 1]);
      });
      if (o.verbose) {
        console.log(`[iter ${iter}] parts=${parts.length} cutting=${cutting.length} worst=${worst.toFixed(4)} gpu=${gpuMs.toFixed(0)}ms cpu=${cpuMs.toFixed(0)}ms`);
      }
      report("iteration", iter, parts.length, cutting.length, worst, base + span, `iteration ${iter} done: ${parts.length} parts`);
    }
    await this.fillHausdorff(parts);
    let merge: MergeStats | null = null;
    let mergeMs = 0;
    if (o.mergeHulls && parts.length > 1) {
      checkAbort();
      const tm0 = performance.now();
      const mergeBase = Math.max(lastFrac, 0.9);
      report("merge", iterations, parts.length, 0, 0, mergeBase, `merging hulls: ${parts.length} parts`);
      const r = await mergeHulls(this.tree, parts, {
        threshold: o.threshold, maxRounds: o.mergeRounds, planeTol: o.mergePlaneTol,
        hullBackend: o.hullBackend, hausdorffBackend: o.hausdorffBackend, verbose: o.verbose, signal: o.signal, hullEps: this.hullEps,
        onRound: (ri) => report("merge", iterations, ri.nParts, 0, 0, mergeBase + (1 - mergeBase) * (1 - Math.pow(0.5, ri.round + 1)), `merge round ${ri.round}: ${ri.nParts} parts, ${ri.candidates} candidate pairs${ri.merged >= 0 ? `, ${ri.merged} merged` : ""}`),
      });
      parts = r.parts;
      merge = r.stats;
      mergeMs = performance.now() - tm0;
      if (o.verbose) console.log(`[merge] ${r.stats.merges} merges in ${r.stats.rounds} rounds → ${parts.length} parts (${mergeMs.toFixed(0)} ms)`);
    }
    report("done", iterations, parts.length, 0, 0, 1, `done: ${parts.length} parts`);
    const out = norm
      ? parts.map((p) => ({
        ...p,
        mesh: denormalize(p.mesh, norm!),
        hull: denormalize(p.hull, norm!),
        seams: p.seams?.map((q) => denormalizePlane(q, norm!)),
      }))
      : parts;
    return {
      parts: out, iterations,
      stats: { gpuMs, cpuMs, totalMs: performance.now() - tStart, treeRuns, hullFailRuns, hausdorffMs, cutMs, concaveMs, hullMs, treeMs, mergeMs, kernelMs },
      merge,
      norm,
    };
  }
}

/** Plane (a, b, c, d), a x + b y + c z + d = 0 in normalized space, expressed in input units. */
function denormalizePlane(p: Plane, n: Normalization): Plane {
  return [p[0], p[1], p[2], p[3] * n.scale - (p[0] * n.center[0] + p[1] * n.center[1] + p[2] * n.center[2])];
}

/** Convenience: acquire a device, decompose, release. */

export async function decompose(mesh: Mesh, opts: Partial<DecomposeOptions> = {}): Promise<DecomposeResult> {
  const { device } = await getDevice();
  const d = new Decomposer(device, opts);
  try {
    return await d.decompose(mesh);
  } finally {
    d.destroy();
  }
}
