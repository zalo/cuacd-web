// Output hulls must be clean meshes: closed, outward, convex within a small
// tolerance of the part's extent, and free of degenerate faces (the exact
// grid hull and the CPU quickhull both produce needle faces on their own).
import { assert } from "@std/assert";
import { getDevice } from "../src/gpu/device.ts";
import { Decomposer } from "../src/decompose.ts";
import { isClosed, type Mesh, signedVolume } from "../src/mesh.ts";
import { loadSTL, stairs } from "./fixtures.ts";

const { device } = await getDevice();

/** issues with a hull of the given points, or [] */
export function hullIssues(h: Mesh, pts: Mesh): string[] {
  const V = h.verts, T = h.tris;
  const issues: string[] = [];
  if (!isClosed(h)) issues.push("not closed");
  if (!(signedVolume(h) > 0)) issues.push("not outward");
  let lo = Infinity, hi = -Infinity;
  for (const x of pts.verts) { lo = Math.min(lo, x); hi = Math.max(hi, x); }
  const extent = hi - lo;
  let maxOut = 0, degenerate = 0;
  for (let t = 0; t < T.length; t += 3) {
    const a = T[t] * 3, b = T[t + 1] * 3, c = T[t + 2] * 3;
    const ux = V[b] - V[a], uy = V[b + 1] - V[a + 1], uz = V[b + 2] - V[a + 2];
    const wx = V[c] - V[a], wy = V[c + 1] - V[a + 1], wz = V[c + 2] - V[a + 2];
    let nx = uy * wz - uz * wy, ny = uz * wx - ux * wz, nz = ux * wy - uy * wx;
    const l = Math.hypot(nx, ny, nz);
    const lab = Math.hypot(ux, uy, uz), lac = Math.hypot(wx, wy, wz);
    if (!(l > 1e-6 * lab * lac)) { degenerate++; continue; }
    nx /= l; ny /= l; nz /= l;
    const d = nx * V[a] + ny * V[a + 1] + nz * V[a + 2];
    const P = pts.verts;
    for (let k = 0; k < P.length; k += 3) maxOut = Math.max(maxOut, nx * P[k] + ny * P[k + 1] + nz * P[k + 2] - d);
  }
  if (degenerate) issues.push(`${degenerate} degenerate faces`);
  if (maxOut > 1e-4 * extent) issues.push(`points outside by ${(maxOut / extent).toExponential(2)} of the extent`);
  return issues;
}

async function checkAll(mesh: Mesh, opts: Record<string, unknown>, label: string) {
  const d = new Decomposer(device, { maxIters: 100, threshold: 0.05, ...opts });
  const r = await d.decompose(mesh);
  d.destroy();
  const bad: string[] = [];
  r.parts.forEach((p, i) => { const iss = hullIssues(p.hull, p.mesh); if (iss.length) bad.push(`part ${i}: ${iss.join(", ")}`); });
  assert(bad.length === 0, `${label}: ${bad.length} of ${r.parts.length} hulls bad\n${bad.slice(0, 5).join("\n")}`);
  return r.parts.length;
}

Deno.test("output hulls: staircase, with and without merging", async () => {
  await checkAll(stairs(), { mergeHulls: false }, "stairs");
  await checkAll(stairs(), { mergeHulls: true }, "stairs merged");
});

Deno.test("output hulls: 49160.stl, with and without merging", async () => {
  const mesh = await loadSTL(new URL("./data/49160.stl", import.meta.url).pathname);
  const n = await checkAll(mesh, { mergeHulls: false }, "49160");
  const m = await checkAll(mesh, { mergeHulls: true }, "49160 merged");
  assert(m <= n, `merging increased the part count ${n} -> ${m}`);
});
