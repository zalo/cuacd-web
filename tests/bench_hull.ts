// Hull kernel throughput: deno run --allow-all tests/bench_hull.ts [mesh.obj]
// Times GpuTree.hullMeshes on K copies of cut-down pieces of the mesh, per
// kernel stage (1 = k-DOP seeds only, 2 = + seed insertion, 3 = + initial
// assignment, 0 = full hull).
import { getDevice } from "../src/gpu/device.ts";
import { GpuTree } from "../src/gpu/tree.ts";
import { planeCut } from "../src/cpu/cut.ts";
import { type Mesh, nv, parseOBJ, normalize, bbox } from "../src/mesh.ts";

const file = Deno.args[0] ?? "/tmp/meshes/bunny_vox64.obj";
const { device } = await getDevice();
const mesh = normalize(parseOBJ(await Deno.readTextFile(file))).mesh;
const pieces: Mesh[] = [mesh];
// halve repeatedly along the longest axis to get pieces of decreasing size
for (let k = 0; k < 7; k++) {
  const m = pieces[pieces.length - 1];
  const { lo, hi } = bbox(m);
  const ext = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
  const ax = ext.indexOf(Math.max(...ext));
  const n: [number, number, number] = [0, 0, 0]; n[ax] = 1;
  const r = planeCut(m, [n[0], n[1], n[2], -(lo[ax] + hi[ax]) / 2]);
  if (!r.ok || !r.pos || !r.neg) break;
  pieces.push(nv(r.pos) >= nv(r.neg) ? r.pos : r.neg);
}
const tree = new GpuTree(device, { width: 30, width2: 3 });
tree.profile = true;
console.log(`piece sizes: ${pieces.map(nv).join(", ")}`);
console.log("nv      K     stage1   stage2   stage3   full     (GPU us per hull, K in flight)   hullVerts");
for (const m of pieces) {
  const n = nv(m);
  const K = Math.max(64, Math.min(2048, Math.floor((40 << 20) / (3 * n))));
  const copies = new Array(K).fill({ verts: m.verts, tris: new Uint32Array(0) });
  const row: string[] = [];
  let hv = 0;
  for (const stage of [1, 2, 3, 0]) {
    await tree.hullMeshes(copies, { hullStage: stage, noExport: true }); // warm-up
    await tree.hullMeshes(copies, { hullStage: stage, noExport: true });
    const dt = tree.lastHullGpuMs;
    row.push(((dt * 1000) / K).toFixed(2).padStart(8));
    if (stage === 0) { const res = await tree.hullMeshes([copies[0]]); if (res[0]) hv = new Set(res[0].tris).size; }
  }
  console.log(`${String(n).padStart(6)} ${String(K).padStart(5)} ${row.join(" ")}   ${hv}`);
}
tree.destroy();
