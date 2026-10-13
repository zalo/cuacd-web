// hull.ts — convex hull of a point set on the CPU (double precision).
//
// The default backend is the Emscripten build of Antti Kuukka's quickhull
// (wasm/qh.cpp → wasm/qh.js, a single-file ES module). Call `initHull()`
// once before `convexHull`; `setHullBackend` swaps in another implementation.
// Results are outward-wound triangles over a compacted vertex list.

import type { Mesh } from "../mesh.ts";
import { signedVolume, isClosed } from "../mesh.ts";

export interface HullResult {
  mesh: Mesh;
  volume: number;
}

export interface HullBackend {
  name: string;
  /** eps is relative to the largest coordinate magnitude (Kuukka's convention); default 1e-7 */
  compute(points: Float64Array, n: number, eps?: number): { verts: Float64Array; tris: Uint32Array };
}

let backend: HullBackend | null = null;

export function setHullBackend(b: HullBackend | null) {
  backend = b;
}
export function hullBackendName(): string {
  return backend?.name ?? "none";
}

let initPromise: Promise<boolean> | null = null;

/** Load the Emscripten quickhull (wasm/qh.js) as the active backend (idempotent). */
export function initHull(): Promise<boolean> {
  if (backend) return Promise.resolve(true);
  if (!initPromise) initPromise = loadWasm();
  return initPromise;
}

async function loadWasm(): Promise<boolean> {
  try {
    // @ts-ignore: generated Emscripten module without type declarations
    const mod: any = await import("../../wasm/qh.js");
    const M: any = await mod.default();
    const wasm: HullBackend = {
      name: "kuukka-wasm",
      compute(points, n, eps = 1e-7) {
        const ptr = M._qh_alloc(n * 3);
        M.HEAPF64.set(points.subarray(0, n * 3), ptr / 8);
        const ntri = M._qh_compute(ptr, n, eps); // scaled by the max coordinate internally
        M._qh_free(ptr);
        const nvOut = M._qh_nv();
        const verts = new Float64Array(M.HEAPF64.buffer, M._qh_verts(), nvOut * 3).slice();
        const tris = new Uint32Array(M.HEAPU32.buffer, M._qh_tris(), ntri * 3).slice();
        return { verts, tris };
      },
    };
    backend = wasm;
    return true;
  } catch (_e) {
    return false;
  }
}

/** Convex hull of the vertices of a mesh or point array; triangles wound outward. */
export function convexHull(verts: Float32Array | Float64Array): HullResult {
  const n = verts.length / 3;
  const pts = verts instanceof Float64Array ? verts : Float64Array.from(verts);
  if (n < 4) return { mesh: { verts: Float32Array.from(pts), tris: new Uint32Array(0) }, volume: 0 };
  if (!backend) throw new Error("hull backend not initialised: await initHull() first");
  const { verts: hv, tris } = backend.compute(pts, n);
  return compactHull(hv, tris);
}

/** Build an outward-wound hull mesh from triangles indexing `hv`, dropping unused vertices. */
/**
 * Clean the faces of an exact (grid-quantized) hull for output. The grid
 * makes points that are collinear or coplanar in float coordinates into hull
 * vertices, so the raw export carries needle faces and extra vertices. The
 * hull is rebuilt in double precision from the raw hull's own vertex set (a
 * few hundred well-spread points, which the CPU quickhull handles reliably)
 * and accepted only when it is closed, keeps the volume and contains every
 * raw vertex; otherwise the raw faces are returned. `eps` (relative to the
 * extent) must stay well above the float32 resolution of the coordinates the
 * hull will be stored in: faces thinner than that get unreliable normals.
 */
export function cleanHullFaces(V: Float32Array | Float64Array, T: Uint32Array, eps = 2e-5): Uint32Array {
  if (!backend || T.length < 12) return T;
  let cand: number[] = [];
  const seen = new Set<number>();
  for (let i = 0; i < T.length; i++) if (!seen.has(T[i])) { seen.add(T[i]); cand.push(T[i]); }
  const vRaw = Math.abs(signedVolume({ verts: V as Float32Array, tris: T }));
  let lo = Infinity, hi = -Infinity;
  for (const c of cand) for (let a = 0; a < 3; a++) { lo = Math.min(lo, V[3 * c + a]); hi = Math.max(hi, V[3 * c + a]); }
  const extent = Math.max(hi - lo, 1e-30);
  const tol = 5 * eps * extent;
  // The CPU hull emits exactly collinear triangles for points on hull edges;
  // such a point is dropped from the candidates and the hull rebuilt.
  for (let round = 0; round < 6; round++) {
    const n = cand.length;
    if (n < 4) return T;
    const pts = new Float64Array(3 * n);
    const byKey = new Map<string, number>();
    for (let k = 0; k < n; k++) {
      const v = cand[k] * 3;
      pts[3 * k] = V[v]; pts[3 * k + 1] = V[v + 1]; pts[3 * k + 2] = V[v + 2];
      byKey.set(`${V[v]},${V[v + 1]},${V[v + 2]}`, cand[k]);
    }
    // an epsilon well above float32 resolution (about 1e-7 relative), so no
    // accepted face is a needle with an unreliable normal once the
    // coordinates are stored or rescaled as float32
    const { verts: hv, tris } = backend.compute(pts, n, eps);
    const out = new Uint32Array(tris.length);
    for (let i = 0; i < tris.length; i++) {
      const h = tris[i] * 3;
      const idx = byKey.get(`${hv[h]},${hv[h + 1]},${hv[h + 2]}`);
      if (idx === undefined) return T;
      out[i] = idx;
    }
    const mesh: Mesh = { verts: V as Float32Array, tris: out };
    if (!isClosed(mesh)) return T;
    const svNew = signedVolume(mesh);
    if (svNew < 0) { // the backend winds inward: orient outward before the containment test
      for (let i = 0; i < out.length; i += 3) { const t = out[i + 1]; out[i + 1] = out[i + 2]; out[i + 2] = t; }
    }
    if (!(Math.abs(Math.abs(svNew) - vRaw) <= 10 * eps * vRaw + 1e-30)) return T;
    // degenerate faces: remove the vertex lying between the other two
    const drop = new Set<number>();
    let ok = true;
    for (let t = 0; t < out.length && ok; t += 3) {
      const ia = out[t], ib = out[t + 1], ic = out[t + 2];
      const a = ia * 3, b = ib * 3, c = ic * 3;
      const ux = V[b] - V[a], uy = V[b + 1] - V[a + 1], uz = V[b + 2] - V[a + 2];
      const wx = V[c] - V[a], wy = V[c + 1] - V[a + 1], wz = V[c + 2] - V[a + 2];
      let nx = uy * wz - uz * wy, ny = uz * wx - ux * wz, nz = ux * wy - uy * wx;
      const l = Math.hypot(nx, ny, nz);
      const lab = Math.hypot(ux, uy, uz), lac = Math.hypot(wx, wy, wz), lbc = Math.hypot(V[c] - V[b], V[c + 1] - V[b + 1], V[c + 2] - V[b + 2]);
      const longest = Math.max(lab, lac, lbc);
      // thin face (height below the tolerance): the vertex opposite the
      // longest edge lies within that height of the hull, so it is dropped
      if (!(l > eps * extent * longest)) {
        drop.add(lbc >= lab && lbc >= lac ? ia : lac >= lab ? ib : ic);
        continue;
      }
      nx /= l; ny /= l; nz /= l;
      const d = nx * V[a] + ny * V[a + 1] + nz * V[a + 2];
      for (const q of cand) { if (nx * V[3 * q] + ny * V[3 * q + 1] + nz * V[3 * q + 2] - d > tol) { ok = false; break; } }
    }
    if (!ok) return T;
    if (drop.size === 0) return out;
    cand = cand.filter((c) => !drop.has(c));
  }
  return T;
}

export function compactHull(hv: Float32Array | Float64Array, tris: Uint32Array): HullResult {
  // compact + orient
  const remap = new Map<number, number>();
  const out: number[] = [];
  const idx = new Uint32Array(tris.length);
  let cx = 0, cy = 0, cz = 0;
  for (let i = 0; i < tris.length; i++) {
    const v = tris[i];
    let r = remap.get(v);
    if (r === undefined) {
      r = out.length / 3;
      remap.set(v, r);
      out.push(hv[3 * v], hv[3 * v + 1], hv[3 * v + 2]);
      cx += hv[3 * v]; cy += hv[3 * v + 1]; cz += hv[3 * v + 2];
    }
    idx[i] = r;
  }
  const nvOut = out.length / 3;
  if (nvOut > 0) { cx /= nvOut; cy /= nvOut; cz /= nvOut; }
  const vf = new Float32Array(out);
  // Both backends wind consistently; orient the whole surface outward by the
  // sign of its volume (a per-face centroid test flips sliver faces wrongly).
  const mesh: Mesh = { verts: vf, tris: idx };
  const sv = signedVolume(mesh);
  if (sv < 0) {
    for (let i = 0; i < idx.length; i += 3) { const t = idx[i + 1]; idx[i + 1] = idx[i + 2]; idx[i + 2] = t; }
  }
  void cx; void cy; void cz;
  return { mesh, volume: Math.abs(sv) };
}
