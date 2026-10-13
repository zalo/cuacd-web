import { assert, assertAlmostEquals, assertEquals } from "@std/assert";
import { getDevice } from "../src/gpu/device.ts";
import { GpuTree } from "../src/gpu/tree.ts";
import { lShape, makeBox, normalize, unitCube, volume } from "../src/mesh.ts";

const { device } = await getDevice();

Deno.test("cube: depth-1 midpoint cuts give exact halves and hulls", async () => {
  const tree = new GpuTree(device, { width: 3, width2: 3, depth: 2, nConcaveEdges: 0, minEdgeDist: 0.01,
    vertexCap: 1 << 20, triCap: 1 << 20, edgeCap: 1 << 16, scratchWords: 1 << 20, hullBudgetBytes: 8 << 20, hullMaxVerts: 256 });
  const cube = unitCube();
  const res = await tree.run([cube], [], { depth: 1 });
  assertEquals(res.overflow, false);
  assertEquals(res.hullFail, false);
  assertEquals(res.stats.jobsPerLevel[0], 3 + tree.planeSlots); // axis cuts first, then (unused) plane slots
  for (let j = 0; j < 3; j++) {
    const it = res.levels[1][j];
    assertEquals(it.flags, 1, `item ${j} valid`);
    assertEquals(it.np, 2);
    for (let p = 0; p < 2; p++) {
      const part = await tree.readPart(it.partIds[p]);
      assertAlmostEquals(part.meshVol, 0.5, 1e-5, `mesh vol cut ${j} part ${p}`);
      assertAlmostEquals(part.hullVol, 0.5, 1e-4, `hull vol cut ${j} part ${p}`);
      assertEquals(part.nPlanes, 1);
    }
    assertAlmostEquals(it.levelCosts[0], 0, 1e-4);
  }
  // every axis cut is equally good: best cost 0
  assertAlmostEquals(res.seeds[0].bestCost, 0, 1e-4);
  tree.destroy();
});

Deno.test("cube: depth-2 cuts (including parallel planes) keep exact volumes", async () => {
  const tree = new GpuTree(device, { width: 3, width2: 3, depth: 2, nConcaveEdges: 0, minEdgeDist: 0.01,
    vertexCap: 1 << 20, triCap: 1 << 20, edgeCap: 1 << 16, scratchWords: 1 << 20, hullBudgetBytes: 8 << 20, hullMaxVerts: 256 });
  const res = await tree.run([unitCube()], [], { depth: 2 });
  assertEquals(res.overflow, false);
  assertEquals(res.hullFail, false);
  assertEquals(res.stats.jobsPerLevel[1], (3 + tree.planeSlots) * 3);
  let checked = 0;
  for (let j = 0; j < 9; j++) {
    const it = res.levels[2][j];
    assertEquals(it.flags, 1, `level-2 item ${j} valid`);
    assertEquals(it.np, 3);
    // parts sorted by rv: all zero → any order; the two newest are quarters, the inherited one a half
    const vols: number[] = [];
    for (let p = 0; p < 3; p++) {
      const part = await tree.readPart(it.partIds[p]);
      vols.push(part.meshVol);
      assertAlmostEquals(part.hullVol, part.meshVol, 2e-4, `hull==mesh for item ${j} part ${p}`);
      checked++;
    }
    vols.sort();
    assertAlmostEquals(vols[0], 0.25, 1e-5, `quarter ${j}`);
    assertAlmostEquals(vols[1], 0.25, 1e-5, `quarter ${j}`);
    assertAlmostEquals(vols[2], 0.5, 1e-5, `half ${j}`);
  }
  assert(checked === 27);
  tree.destroy();
});

Deno.test("L-shape: best level-0 cut is the x=1 plane", async () => {
  const tree = new GpuTree(device, { width: 9, width2: 3, depth: 2, nConcaveEdges: 0, minEdgeDist: 0.0125,
    vertexCap: 1 << 20, triCap: 1 << 20, edgeCap: 1 << 16, scratchWords: 1 << 20, hullBudgetBytes: 8 << 20, hullMaxVerts: 256 });
  const { mesh } = normalize(lShape());
  const res = await tree.run([mesh], [], { depth: 2 });
  assertEquals(res.overflow, false);
  assertEquals(res.hullFail, false);
  const s = res.seeds[0];
  assert(s.plane !== null);
  // normalized L: x in [-1,1], z in [-0.5,0.5]. Both the x=0 plane and the
  // z=0 plane split it into two boxes, so either zero-cost cut is correct.
  const pl = s.plane!;
  const isX = Math.abs(pl[0] - 1) < 1e-6 && Math.abs(pl[3]) < 1e-3;
  const isZ = Math.abs(pl[2] - 1) < 1e-6 && Math.abs(pl[3]) < 1e-3;
  assert(isX || isZ, `unexpected plane ${pl}`);
  assertAlmostEquals(s.bestCost, 0, 1e-3);
  tree.destroy();
});

Deno.test("off-center box: hull volume matches mesh volume for every candidate", async () => {
  const tree = new GpuTree(device, { width: 6, width2: 3, depth: 2, nConcaveEdges: 0, minEdgeDist: 0.01,
    vertexCap: 1 << 20, triCap: 1 << 20, edgeCap: 1 << 16, scratchWords: 1 << 20, hullBudgetBytes: 8 << 20, hullMaxVerts: 256 });
  const box = makeBox([0.2, -0.7, 0.1], [0.9, 0.3, 0.8]);
  const total = volume(box);
  const res = await tree.run([box], [], { depth: 2 });
  for (const it of res.levels[1]) {
    if (it.flags !== 1) continue;
    let sum = 0;
    for (let p = 0; p < it.np; p++) {
      const part = await tree.readPart(it.partIds[p]);
      assertAlmostEquals(part.hullVol, part.meshVol, 1e-4);
      sum += part.meshVol;
    }
    assertAlmostEquals(sum, total, 1e-5);
  }
  for (const it of res.levels[2]) {
    if (it.flags !== 1) continue;
    let sum = 0;
    for (let p = 0; p < it.np; p++) {
      const part = await tree.readPart(it.partIds[p]);
      assertAlmostEquals(part.hullVol, part.meshVol, 1e-4);
      sum += part.meshVol;
    }
    assertAlmostEquals(sum, total, 1e-5);
  }
  tree.destroy();
});
