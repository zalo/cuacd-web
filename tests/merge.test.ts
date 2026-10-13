import { assert, assertAlmostEquals } from "@std/assert";
import { getDevice } from "../src/gpu/device.ts";
import { GpuTree } from "../src/gpu/tree.ts";
import { Decomposer } from "../src/decompose.ts";
import { mergeHulls, seamPlane } from "../src/merge.ts";
import { partCost, type Part } from "../src/part.ts";
import { convexHull, initHull } from "../src/cpu/hull.ts";
import { planeCut } from "../src/cpu/cut.ts";
import { isClosed, lShape, makeBox, type Mesh, unitCube, volume } from "../src/mesh.ts";
import { stairs } from "./fixtures.ts";

const { device } = await getDevice();
await initHull();
const smallTree = { width: 3, width2: 3, vertexCap: 4 << 20, triCap: 4 << 20, edgeCap: 1 << 18, scratchWords: 4 << 20, hullBudgetBytes: 32 << 20, hullMaxVerts: 1024 };

function part(mesh: Mesh): Part {
  const h = convexHull(mesh.verts);
  return { mesh, hull: h.mesh, meshVol: volume(mesh), hullVol: h.volume, hausdorff: 0, stuck: false };
}
function slabs(cuts: number[]): Part[] {
  let rest = unitCube();
  const out: Part[] = [];
  for (const x of cuts) {
    const r = planeCut(rest, [1, 0, 0, -x]);
    assert(r.ok && r.pos && r.neg);
    out.push(part(r.neg!));
    rest = r.pos!;
  }
  out.push(part(rest));
  return out;
}

Deno.test("merge: two halves of a cube become one part", async () => {
  const tree = new GpuTree(device, smallTree);
  const parts = slabs([0.5]);
  const seam = seamPlane(parts[0], parts[1], 1e-3);
  assert(seam && Math.abs(Math.abs(seam[0]) - 1) < 1e-6 && Math.abs(Math.abs(seam[3]) - 0.5) < 1e-6, `seam ${seam}`);
  const r = await mergeHulls(tree, parts, { threshold: 0.05 });
  tree.destroy();
  assert(r.parts.length === 1, `parts=${r.parts.length}`);
  const p = r.parts[0];
  assertAlmostEquals(p.meshVol, 1, 1e-6);
  assertAlmostEquals(p.hullVol, 1, 1e-5);
  assert(partCost(p) < 0.01, `cost ${partCost(p)}`);
  assert(isClosed(p.mesh));
  assert(p.seams?.length === 1 && p.merged === 2);
  assert(r.stats.merges === 1 && r.stats.rounds === 1);
});

Deno.test("merge: CPU backends give the same answer", async () => {
  const r = await mergeHulls(null, slabs([0.5]), { threshold: 0.05, hullBackend: "cpu", hausdorffBackend: "cpu" });
  assert(r.parts.length === 1);
  assertAlmostEquals(r.parts[0].hullVol, 1, 1e-5);
});

Deno.test("merge: the two boxes of an L-shape stay apart", async () => {
  const tree = new GpuTree(device, smallTree);
  const parts = [part(makeBox([0, 0, 0], [1, 1, 1])), part(makeBox([1, 0, 0], [2, 1, 0.5]))];
  const r = await mergeHulls(tree, parts, { threshold: 0.05 });
  tree.destroy();
  assert(r.parts.length === 2);
  assert(r.stats.pairsHulled === 1 && r.stats.merges === 0);
});

Deno.test("merge: four slabs need more than one round", async () => {
  const tree = new GpuTree(device, smallTree);
  const r = await mergeHulls(tree, slabs([0.25, 0.5, 0.75]), { threshold: 0.05 });
  tree.destroy();
  assert(r.parts.length === 1, `parts=${r.parts.length}`);
  assert(r.stats.rounds === 2 && r.stats.merges === 3, JSON.stringify(r.stats));
  assertAlmostEquals(r.parts[0].meshVol, 1, 1e-6);
  assert(r.parts[0].seams!.length === 3);
  // nothing to merge with a single round of the pass
  const one = await mergeHulls(null, slabs([0.25, 0.5, 0.75]), { threshold: 0.05, maxRounds: 1, hullBackend: "cpu", hausdorffBackend: "cpu" });
  assert(one.parts.length === 2);
});

Deno.test("merge: decomposing stairs with merging keeps 3 parts and the volume", async () => {
  const d = new Decomposer(device, { maxIters: 50, threshold: 0.05, width: 30, tree: smallTree, mergeHulls: true });
  const r = await d.decompose(stairs());
  d.destroy();
  assert(r.merge !== null);
  assert(r.parts.length >= 3 && r.parts.length <= 6, `parts=${r.parts.length}`);
  let total = 0;
  for (const p of r.parts) { assert(isClosed(p.mesh)); assert(partCost(p) < 0.05 + 1e-6 || p.stuck); total += volume(p.mesh); }
  assertAlmostEquals(total, 6, 1e-4);
});

Deno.test("merge: L-shape decomposition ends at exactly 2 parts", async () => {
  const d = new Decomposer(device, { maxIters: 50, threshold: 0.05, width: 30, tree: smallTree, mergeHulls: true });
  const r = await d.decompose(lShape());
  d.destroy();
  assert(r.parts.length === 2, `parts=${r.parts.length}`);
  let total = 0;
  for (const p of r.parts) total += volume(p.mesh);
  assertAlmostEquals(total, 1.5, 1e-4);
});
