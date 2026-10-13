// Benchmark: deno run --allow-all tests/bench_decompose.ts [mesh.obj|mesh.stl ...]
import { getDevice } from "../src/gpu/device.ts";
import { Decomposer, partCost } from "../src/decompose.ts";
import { isClosed, nt, nv, parseOBJ, parseSTL, volume } from "../src/mesh.ts";

const noMerge = Deno.args.includes("--no-merge");
const files = Deno.args.filter((a) => !a.startsWith("--"));
if (files.length === 0) files.push("/tmp/meshes/bunny_vox64.obj", "/tmp/meshes/teapot_vox64.obj");
const { device, info } = await getDevice();
console.log(`adapter: ${info.vendor} ${info.description}`);
for (const f of files) {
  let mesh;
  try {
    mesh = f.endsWith(".stl") ? parseSTL((await Deno.readFile(f)).buffer as ArrayBuffer) : parseOBJ(await Deno.readTextFile(f));
  } catch (e) { console.log(`skip ${f}: ${(e as Error).message}`); continue; }
  console.log(`\n${f}: ${nv(mesh)} verts, ${nt(mesh)} tris, closed=${isClosed(mesh)}`);
  const d = new Decomposer(device, { maxIters: 100, threshold: 0.05, width: 30, nConcaveEdges: 16, verbose: 0, profile: true, mergeHulls: !noMerge });
  const r = await d.decompose(mesh);
  d.destroy();
  const vin = volume(mesh);
  let vsum = 0, closed = 0, worst = 0;
  for (const p of r.parts) { vsum += volume(p.mesh); if (isClosed(p.mesh)) closed++; worst = Math.max(worst, partCost(p)); }
  const stuck: Record<string, number> = {};
  for (const p of r.parts) if (p.stuck) stuck[p.stuckReason ?? "?"] = (stuck[p.stuckReason ?? "?"] ?? 0) + 1;
  const st = r.stats;
  const k = Object.entries(st.kernelMs).map(([n, v]) => `${n}=${v.toFixed(0)}`).join(" ");
  console.log(`   breakdown: tree ${st.treeMs.toFixed(0)} ms (kernels: ${k}) | hulls ${st.hullMs.toFixed(0)} | hausdorff ${st.hausdorffMs.toFixed(0)} | cuts ${st.cutMs.toFixed(0)} | concave ${st.concaveMs.toFixed(0)}`);
  if (r.merge) console.log(`   merge: ${r.merge.merges} merges in ${r.merge.rounds} rounds (${r.merge.pairsHulled} hulls, ${r.merge.pairsHausdorff} hausdorff) ${st.mergeMs.toFixed(0)} ms (hull ${r.merge.hullMs.toFixed(0)}, hausdorff ${r.merge.hausdorffMs.toFixed(0)})`);
  console.log(`=> ${r.parts.length} parts, ${r.iterations} iters, total ${r.stats.totalMs.toFixed(0)} ms (gpu ${r.stats.gpuMs.toFixed(0)}, cpu ${r.stats.cpuMs.toFixed(0)}, ${r.stats.treeRuns} tree runs, ${r.stats.hullFailRuns} with hull failures); closed ${closed}/${r.parts.length}; volume ${vsum.toFixed(4)} vs ${vin.toFixed(4)}; worst cost ${worst.toFixed(4)}; stuck ${JSON.stringify(stuck)}`);
}
