import { assert, assertAlmostEquals } from "@std/assert";
import { getDevice } from "../src/gpu/device.ts";
import { Decomposer, partCost } from "../src/decompose.ts";
import { isClosed, lShape, unitCube, volume } from "../src/mesh.ts";
import { loadSTL, squareTorus, stairs } from "./fixtures.ts";

const { device } = await getDevice();
const smallTree = { vertexCap: 4 << 20, triCap: 4 << 20, edgeCap: 1 << 18, scratchWords: 4 << 20, hullBudgetBytes: 32 << 20, hullMaxVerts: 1024 };

Deno.test("decompose: cube stays one part", async () => {
  const d = new Decomposer(device, { maxIters: 5, threshold: 0.05, tree: smallTree });
  const r = await d.decompose(unitCube());
  d.destroy();
  assert(r.parts.length === 1);
  assertAlmostEquals(volume(r.parts[0].mesh), 1, 1e-6);
});

Deno.test("decompose: L-shape becomes 2 convex parts", async () => {
  const d = new Decomposer(device, { maxIters: 50, threshold: 0.05, width: 30, tree: smallTree });
  const r = await d.decompose(lShape());
  d.destroy();
  assert(r.parts.length >= 2 && r.parts.length <= 10, `parts=${r.parts.length}`);
  let total = 0;
  for (const p of r.parts) {
    assert(isClosed(p.mesh));
    assert(partCost(p) < 0.05 + 1e-6 || p.stuck, `cost ${partCost(p)}`);
    total += volume(p.mesh);
  }
  assertAlmostEquals(total, 1.5, 1e-4);
});

Deno.test("decompose: stairs → at least 3 parts, volume preserved", async () => {
  const d = new Decomposer(device, { maxIters: 50, threshold: 0.05, width: 30, tree: smallTree });
  const r = await d.decompose(stairs());
  d.destroy();
  assert(r.parts.length >= 3 && r.parts.length <= 12, `parts=${r.parts.length}`);
  let total = 0;
  for (const p of r.parts) { assert(isClosed(p.mesh)); total += volume(p.mesh); }
  assertAlmostEquals(total, 6, 1e-4);
});

Deno.test("decompose: square torus (genus 1) → closed parts, volume preserved", async () => {
  const d = new Decomposer(device, { maxIters: 50, threshold: 0.05, width: 30, tree: smallTree });
  const r = await d.decompose(squareTorus());
  d.destroy();
  assert(r.parts.length >= 4, `parts=${r.parts.length}`);
  let total = 0;
  for (const p of r.parts) { assert(isClosed(p.mesh)); total += volume(p.mesh); }
  assertAlmostEquals(total, 1.5, 1e-4);
});

Deno.test("decompose: 49160.stl converges with closed parts", async () => {
  const mesh = await loadSTL(new URL("./data/49160.stl", import.meta.url).pathname);
  assert(isClosed(mesh));
  const d = new Decomposer(device, { maxIters: 100, threshold: 0.05, width: 30, tree: smallTree });
  const r = await d.decompose(mesh);
  d.destroy();
  console.log(`49160: ${r.parts.length} parts in ${r.iterations} iterations, gpu ${r.stats.gpuMs.toFixed(0)} ms, cpu ${r.stats.cpuMs.toFixed(0)} ms`);
  assert(r.parts.length >= 1);
  let total = 0;
  for (const p of r.parts) { assert(isClosed(p.mesh)); total += volume(p.mesh); }
  assertAlmostEquals(total, volume(mesh), volume(mesh) * 1e-3);
});
