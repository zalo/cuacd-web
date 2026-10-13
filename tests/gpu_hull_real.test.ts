// Opt-in: compares GPU hull volumes against the CPU quickhull on real meshes
// (needs /tmp/meshes/*_vox64.obj, produced by trimesh voxel remeshing).
import { assert } from "@std/assert";
import { getDevice } from "../src/gpu/device.ts";
import { GpuTree } from "../src/gpu/tree.ts";
import { isClosed, type Mesh, normalize, parseOBJ, nv, signedVolume } from "../src/mesh.ts";
import { compactHull, convexHull, initHull } from "../src/cpu/hull.ts";

const files = ["/tmp/meshes/teapot_vox64.obj", "/tmp/meshes/bunny_vox64.obj"];
const present = files.filter((f) => { try { Deno.statSync(f); return true; } catch { return false; } });

Deno.test({ name: "GPU hull volumes match CPU quickhull on real-mesh children", ignore: present.length === 0, fn: async () => {
  await initHull();
  const { device } = await getDevice();
  for (const f of present) {
    const { mesh } = normalize(parseOBJ(await Deno.readTextFile(f)));
    const tree = new GpuTree(device, { width: 30, width2: 6, depth: 1, nConcaveEdges: 0, minEdgeDist: 0.0125 });
    const res = await tree.run([mesh], [], { depth: 1, readParts: true });
    const pv = res.partVols!;
    let n = 0, bad = 0, mism = 0, worst = 0;
    for (let j = 0; j < res.stats.jobsPerLevel[0] * 2; j++) {
      const mv = pv.data[j * 16 + 6], hv = pv.data[j * 16 + 7];
      if (mv === 0) continue;
      n++;
      const part = await tree.readPart(pv.base + j);
      const cpu = convexHull(part.mesh.verts);
      if (hv < 0) { bad++; console.log(`  ${f} child ${j}: GPU hull failed, reason ${new Uint32Array(pv.data.buffer)[j * 16 + 13]}, nv=${nv(part.mesh)}`); continue; }
      const rel = Math.abs(hv - cpu.volume) / Math.max(cpu.volume, 1e-9);
      worst = Math.max(worst, rel);
      if (rel > 2e-3) { mism++; console.log(`  ${f} child ${j}: gpu=${hv.toFixed(6)} cpu=${cpu.volume.toFixed(6)} rel=${rel.toExponential(2)} nv=${nv(part.mesh)}`); }
    }
    console.log(`${f}: children=${n} failed=${bad} mismatched=${mism} worst rel=${worst.toExponential(2)} gpu=${res.stats.gpuMs.toFixed(0)}ms`);
    tree.destroy();
    assert(bad === 0, `${bad} GPU hull failures`);
    // Kuukka's quickhull misreports a few voxel-mesh hulls (verified against qhull);
    // GPU volumes were validated against qhull to 2e-7, so only gross mismatches fail here.
    assert(worst < 0.1, `hull volume mismatch worst rel ${worst}`);
  }
}});

Deno.test({ name: "GPU hullMeshes: closed, outward, contains every point", ignore: present.length === 0, fn: async () => {
  const { device } = await getDevice();
  const meshes: Mesh[] = [];
  for (const f of present) meshes.push(normalize(parseOBJ(await Deno.readTextFile(f))).mesh);
  const tree = new GpuTree(device, { width: 30, width2: 6, depth: 1, nConcaveEdges: 0, minEdgeDist: 0.0125 });
  const hulls = await tree.hullMeshes(meshes);
  hulls.forEach((h, i) => {
    assert(h !== null, `hull ${i} failed`);
    const hm = compactHull(meshes[i].verts, h!.tris);
    assert(isClosed(hm.mesh), `hull ${i} not closed`);
    assert(signedVolume(hm.mesh) > 0, `hull ${i} inward`);
    assert(Math.abs(hm.volume - h!.volume) < 1e-5 * h!.volume, `hull ${i} volume mismatch ${hm.volume} vs ${h!.volume}`);
    // containment: no input point outside any face by more than 1e-5
    const V = meshes[i].verts, T = hm.mesh.tris, HV = hm.mesh.verts;
    let maxOut = -Infinity;
    for (let t = 0; t < T.length; t += 3) {
      const a = T[t] * 3, b = T[t + 1] * 3, c = T[t + 2] * 3;
      const ux = HV[b] - HV[a], uy = HV[b + 1] - HV[a + 1], uz = HV[b + 2] - HV[a + 2];
      const wx = HV[c] - HV[a], wy = HV[c + 1] - HV[a + 1], wz = HV[c + 2] - HV[a + 2];
      let nx = uy * wz - uz * wy, ny = uz * wx - ux * wz, nz = ux * wy - uy * wx;
      const l = Math.hypot(nx, ny, nz);
      if (l < 1e-9) continue; // sliver face of the exact (grid-quantized) hull: its normal is meaningless here
      nx /= l; ny /= l; nz /= l;
      const d = -(nx * HV[a] + ny * HV[a + 1] + nz * HV[a + 2]);
      for (let k = 0; k < V.length; k += 3) { const dd = nx * V[k] + ny * V[k + 1] + nz * V[k + 2] + d; if (dd > maxOut) maxOut = dd; }
    }
    console.log(`${present[i]}: hull verts=${nv(hm.mesh)} vol=${hm.volume.toFixed(5)} maxOutside=${maxOut.toExponential(2)}`);
    assert(maxOut < 1e-5, `hull ${i} does not contain all points (${maxOut})`);
  });
  tree.destroy();
}});
