import { assert, assertAlmostEquals, assertEquals } from "@std/assert";
import { isClosed, lShape, makeBox, signedVolume, unitCube, volume } from "../src/mesh.ts";
import { planeCut } from "../src/cpu/cut.ts";
import { convexHull, hullBackendName, initHull } from "../src/cpu/hull.ts";
import { hausdorff } from "../src/cpu/hausdorff.ts";
import { concaveEdgePlanes } from "../src/cpu/concave.ts";
import { lPrism, squareTorus, stairs } from "./fixtures.ts";

Deno.test("fixtures are closed and outward wound", () => {
  for (const m of [unitCube(), lShape(), squareTorus(), stairs(), lPrism()]) {
    assert(isClosed(m));
    assert(signedVolume(m) > 0);
  }
  assertAlmostEquals(volume(squareTorus()), 1.5, 1e-6);
  assertAlmostEquals(volume(lShape()), 1.5, 1e-6);
});

Deno.test("planeCut: cube halves are closed with exact volumes", () => {
  const cube = unitCube();
  for (const plane of [[1, 0, 0, -0.5], [0, 1, 0, -0.3], [0, 0, 1, -0.75], [0.6, 0.8, 0, -0.7]] as [number, number, number, number][]) {
    const r = planeCut(cube, plane);
    assert(r.ok, r.reason);
    assert(isClosed(r.pos!) && isClosed(r.neg!));
    assert(signedVolume(r.pos!) > 0 && signedVolume(r.neg!) > 0);
    assertAlmostEquals(volume(r.pos!) + volume(r.neg!), 1, 1e-6);
  }
  const r = planeCut(cube, [1, 0, 0, -0.5]);
  assertAlmostEquals(volume(r.pos!), 0.5, 1e-6);
});

Deno.test("planeCut: on-plane vertices (cube at x=1 face, L junction)", () => {
  const cube = unitCube();
  const r0 = planeCut(cube, [1, 0, 0, -1]);
  assertEquals(r0.ok, false); // nothing strictly positive
  const L = lShape();
  const r = planeCut(L, [1, 0, 0, -1]);
  assert(r.ok, r.reason);
  assert(isClosed(r.pos!) && isClosed(r.neg!));
  assertAlmostEquals(volume(r.pos!), 0.5, 1e-6);
  assertAlmostEquals(volume(r.neg!), 1, 1e-6);
});

Deno.test("planeCut: torus cut yields caps with holes", () => {
  const t = squareTorus();
  const r = planeCut(t, [0, 0, 1, 0]);
  assert(r.ok, r.reason);
  assert(isClosed(r.pos!), "pos closed");
  assert(isClosed(r.neg!), "neg closed");
  assertAlmostEquals(volume(r.pos!), 0.75, 1e-6);
  assertAlmostEquals(volume(r.neg!), 0.75, 1e-6);
  // cut through the ring: two separate loops on each side
  const r2 = planeCut(t, [1, 0, 0, 0]);
  assert(r2.ok, r2.reason);
  assert(isClosed(r2.pos!) && isClosed(r2.neg!));
  assertAlmostEquals(volume(r2.pos!) + volume(r2.neg!), 1.5, 1e-6);
});

Deno.test("convexHull (wasm quickhull): exact on boxes, closed on random clouds", async () => {
  const ok = await initHull();
  assert(ok, "wasm hull failed to load");
  assertEquals(hullBackendName(), "kuukka-wasm");
  const cube = unitCube();
  assertAlmostEquals(convexHull(cube.verts).volume, 1, 1e-9);
  assertAlmostEquals(convexHull(lShape().verts).volume, 1.75, 1e-9);
  const pts = new Float32Array(3000);
  let s = 1;
  const rnd = () => (s = (s * 16807) % 2147483647) / 2147483647;
  for (let i = 0; i < pts.length; i++) pts[i] = rnd() * 2 - 1;
  const h = convexHull(pts);
  assert(isClosed(h.mesh));
  assert(signedVolume(h.mesh) > 0);
  assert(h.volume > 6.5 && h.volume < 8, `vol=${h.volume}`);
  // every input point lies inside (or on) the hull
  const hd = hausdorff({ verts: pts, tris: new Uint32Array(0) }, h.mesh);
  assert(hd === 0);
});

Deno.test("hausdorff: convex mesh vs its hull is zero, L-shape is not", async () => {
  await initHull();
  const cube = unitCube();
  assertAlmostEquals(hausdorff(cube, convexHull(cube.verts).mesh), 0, 1e-6);
  const L = lShape();
  const hd = hausdorff(L, convexHull(L.verts).mesh);
  assert(hd > 0.3 && hd < 0.6, `hd=${hd}`);
  const a = makeBox([0, 0, 0], [1, 1, 1]), b = makeBox([0.1, 0, 0], [1.1, 1, 1]);
  assertAlmostEquals(hausdorff(a, b), 0.1, 0.02);
});

Deno.test("concaveEdgePlanes: L prism has one concave edge → 4 planes", () => {
  const planes = concaveEdgePlanes(lPrism(), { nConcaveEdges: 8, concaveEps: 0.005, concaveThreshold: 3.49 });
  assertEquals(planes.length, 16);
  // the unwelded L-shape has no shared junction edge, hence no concave edge
  assertEquals(concaveEdgePlanes(lShape(), { nConcaveEdges: 8, concaveEps: 0.005, concaveThreshold: 3.49 }).length, 0);
  assertEquals(concaveEdgePlanes(unitCube(), { nConcaveEdges: 8, concaveEps: 0.005, concaveThreshold: 3.49 }).length, 0);
});

Deno.test("GPU hausdorff matches the CPU implementation", async () => {
  const { getDevice } = await import("../src/gpu/device.ts");
  const { GpuTree } = await import("../src/gpu/tree.ts");
  await initHull();
  const { device } = await getDevice();
  const tree = new GpuTree(device, { width: 3, width2: 3, depth: 1, nConcaveEdges: 0 });
  const L = lShape(), T = squareTorus(), S = stairs();
  const pairs = [L, T, S].map((m) => ({ a: m, b: convexHull(m.verts).mesh }));
  pairs.push({ a: unitCube(), b: makeBox([0.1, 0, 0], [1.1, 1, 1]) });
  const gpu = await tree.hausdorff(pairs);
  pairs.forEach((p, i) => {
    const cpu = hausdorff(p.a, p.b);
    assertAlmostEquals(gpu[i], cpu, 1e-4 + 1e-3 * cpu, `pair ${i}: gpu=${gpu[i]} cpu=${cpu}`);
  });
  tree.destroy();
});
