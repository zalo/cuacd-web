// cut.ts — CPU plane cut with cap triangulation (used for the persistent
// decomposition only; tree candidates are cut uncapped on the GPU).
//
// Mirrors the semantics of cuda/plane_cut.cuh: vertices within PC_EPS of the
// plane count as on-plane, intersection vertices are deduplicated per edge,
// boundary loops are classified by nesting parity (even depth = outer, odd =
// hole) and triangulated with earcut. Any failure to close a loop returns the
// mesh unsplit so the caller can treat the cut as failed.

import earcut from "earcut";
import type { Mesh } from "../mesh.ts";

export const PC_EPS = 1e-6;
export type Plane = [number, number, number, number];

export interface CutResult {
  ok: boolean;
  pos: Mesh | null;
  neg: Mesh | null;
  reason?: string;
}

interface Side {
  tris: number[]; // indices into the combined vertex list
}

export interface CapDebug {
  loops: { len: number; area: number; depth: number; parent: number; tris: number }[];
}
/** Filled by the last planeCut call when `debug` is passed. */
export const lastCutDebug: { pos: CapDebug | null; neg: CapDebug | null } = { pos: null, neg: null };

export function planeCut(mesh: Mesh, plane: Plane, eps = PC_EPS, debug = false): CutResult {
  const V = mesh.verts, T = mesh.tris;
  const nv = V.length / 3, nt = T.length / 3;
  const [a, b, c, d] = plane;
  const nlen = Math.hypot(a, b, c) || 1;
  const n: [number, number, number] = [a / nlen, b / nlen, c / nlen];
  const dd = d / nlen;
  const dist = new Float64Array(nv);
  const sign = new Int8Array(nv);
  let anyPos = false, anyNeg = false;
  for (let i = 0; i < nv; i++) {
    const v = n[0] * V[3 * i] + n[1] * V[3 * i + 1] + n[2] * V[3 * i + 2] + dd;
    dist[i] = v;
    sign[i] = v > eps ? 1 : v < -eps ? -1 : 0;
    if (sign[i] > 0) anyPos = true;
    if (sign[i] < 0) anyNeg = true;
  }
  if (!anyPos || !anyNeg) return { ok: false, pos: null, neg: null, reason: "plane does not split" };

  // combined vertex list: originals + intersection points
  const xyz: number[] = Array.from(V);
  const isect = new Map<number, number>();
  const isectIdx = (i: number, j: number): number => {
    const lo = Math.min(i, j), hi = Math.max(i, j);
    const key = lo * 4294967296 + hi;
    let idx = isect.get(key);
    if (idx !== undefined) return idx;
    const t = dist[lo] / (dist[lo] - dist[hi]);
    idx = xyz.length / 3;
    xyz.push(
      V[3 * lo] + t * (V[3 * hi] - V[3 * lo]),
      V[3 * lo + 1] + t * (V[3 * hi + 1] - V[3 * lo + 1]),
      V[3 * lo + 2] + t * (V[3 * hi + 2] - V[3 * lo + 2]),
    );
    isect.set(key, idx);
    return idx;
  };

  const pos: Side = { tris: [] }, neg: Side = { tris: [] };
  const poly: number[] = [];
  const ids = [0, 0, 0];
  const clip = (side: Side, keepSign: number, i0: number, i1: number, i2: number) => {
    // Sutherland–Hodgman on one triangle; on-plane vertices are kept by both sides.
    poly.length = 0;
    ids[0] = i0; ids[1] = i1; ids[2] = i2;
    for (let k = 0; k < 3; k++) {
      const p = ids[k], q = ids[(k + 1) % 3];
      const sp = sign[p] * keepSign, sq = sign[q] * keepSign;
      if (sp >= 0) poly.push(p);
      if (sign[p] !== 0 && sign[q] !== 0 && sp * sq < 0) poly.push(isectIdx(p, q));
    }
    for (let k = 1; k + 1 < poly.length; k++) side.tris.push(poly[0], poly[k], poly[k + 1]);
  };
  for (let t = 0; t < nt; t++) {
    const i0 = T[3 * t], i1 = T[3 * t + 1], i2 = T[3 * t + 2];
    const s0 = sign[i0], s1 = sign[i1], s2 = sign[i2];
    const hasPos = s0 > 0 || s1 > 0 || s2 > 0;
    const hasNeg = s0 < 0 || s1 < 0 || s2 < 0;
    // Triangles lying in the plane are dropped: the side they belong to gets
    // them back from its cap, and keeping them would double the cap.
    if (!hasPos && !hasNeg) continue;
    if (!hasNeg) pos.tris.push(i0, i1, i2);
    else if (!hasPos) neg.tris.push(i0, i1, i2);
    else {
      clip(pos, 1, i0, i1, i2);
      clip(neg, -1, i0, i1, i2);
    }
  }

  const capPos = cap(xyz, pos.tris, n, -1, debug ? (lastCutDebug.pos = { loops: [] }) : null);
  if (!capPos.ok) return { ok: false, pos: null, neg: null, reason: "pos cap: " + capPos.reason };
  const capNeg = cap(xyz, neg.tris, n, +1, debug ? (lastCutDebug.neg = { loops: [] }) : null);
  if (!capNeg.ok) return { ok: false, pos: null, neg: null, reason: "neg cap: " + capNeg.reason };
  return { ok: true, pos: compact(xyz, pos.tris), neg: compact(xyz, neg.tris) };
}

/** Triangulate the open boundary loops of `tris` and append the cap triangles. */
function cap(xyz: number[], tris: number[], n: [number, number, number], capDir: number, dbg: CapDebug | null = null): { ok: boolean; reason?: string } {
  // boundary edges = directed edges without a reverse twin
  const edges = new Map<number, number>();
  const key = (a: number, b: number) => a * 4294967296 + b;
  for (let i = 0; i < tris.length; i += 3) {
    for (let e = 0; e < 3; e++) edges.set(key(tris[i + e], tris[i + ((e + 1) % 3)]), 1);
  }
  const next = new Map<number, number[]>();
  let nBoundary = 0;
  for (const k of edges.keys()) {
    const a = Math.floor(k / 4294967296), b = k % 4294967296;
    if (!edges.has(key(b, a))) {
      nBoundary++;
      let arr = next.get(a);
      if (!arr) { arr = []; next.set(a, arr); }
      arr.push(b);
    }
  }
  if (nBoundary === 0) return { ok: true };
  // walk loops
  const loops: number[][] = [];
  const used = new Set<number>();
  for (const [start, outs] of next) {
    for (const first of outs) {
      if (used.has(key(start, first))) continue;
      const loop = [start];
      let prev = start, cur = first;
      used.add(key(start, first));
      let guard = 0;
      while (cur !== start) {
        loop.push(cur);
        const cands = next.get(cur);
        if (!cands) return { ok: false, reason: "open boundary chain" };
        let nxt = -1;
        for (const cnd of cands) if (!used.has(key(cur, cnd))) { nxt = cnd; break; }
        if (nxt < 0) return { ok: false, reason: "boundary chain dead end" };
        used.add(key(cur, nxt));
        prev = cur; cur = nxt;
        if (++guard > nBoundary + 1) return { ok: false, reason: "boundary chain cycle" };
      }
      void prev;
      if (loop.length >= 3) loops.push(loop);
    }
  }
  if (loops.length === 0) return { ok: true };
  // 2D basis in the plane
  const ax = Math.abs(n[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  const u = cross(ax as [number, number, number], n); normalizeInPlace(u);
  const v = cross(n, u);
  const project = (i: number): [number, number] => {
    const x = xyz[3 * i], y = xyz[3 * i + 1], z = xyz[3 * i + 2];
    return [x * u[0] + y * u[1] + z * u[2], x * v[0] + y * v[1] + z * v[2]];
  };
  const loops2d = loops.map((lp) => lp.map(project));
  const areas = loops2d.map(signedArea);
  // Loop classification. On a consistently wound surface every outer loop of
  // the cap winds one way and every hole the other way, so parity comes from
  // the sign of the signed area relative to the largest loop (always an
  // outer). Point-in-polygon tests are only used to pick a hole's parent, on
  // the hole's area centroid (robust when a hole touches its outer at a
  // vertex, where testing a single vertex is ambiguous).
  const depth = loops.map(() => 0);
  const parent = loops.map(() => -1);
  let largest = -1;
  for (let i = 0; i < loops.length; i++) if (largest < 0 || Math.abs(areas[i]) > Math.abs(areas[largest])) largest = i;
  const outerSign = largest >= 0 ? Math.sign(areas[largest]) : 1;
  for (let i = 0; i < loops.length; i++) {
    if (Math.abs(areas[i]) < 1e-18) continue;
    if (Math.sign(areas[i]) === outerSign) continue; // outer
    depth[i] = 1;
    const c = centroid2d(loops2d[i], areas[i]);
    let bestArea = Infinity, fallback = -1, fallbackArea = -Infinity;
    for (let j = 0; j < loops.length; j++) {
      if (i === j || Math.abs(areas[j]) < 1e-18 || Math.sign(areas[j]) !== outerSign) continue;
      if (Math.abs(areas[j]) > fallbackArea) { fallbackArea = Math.abs(areas[j]); fallback = j; }
      if (Math.abs(areas[j]) <= Math.abs(areas[i])) continue;
      if (pointInPolygon(c, loops2d[j]) && Math.abs(areas[j]) < bestArea) { bestArea = Math.abs(areas[j]); parent[i] = j; }
    }
    if (parent[i] < 0) parent[i] = fallback;
  }
  if (dbg) for (let i = 0; i < loops.length; i++) dbg.loops.push({ len: loops[i].length, area: areas[i], depth: depth[i], parent: parent[i], tris: 0 });
  // zero-area loops (back-and-forth chains along internal flaps or earlier
  // fan caps) cannot be triangulated by earcut; a centroid fan twins their
  // edges with zero-area triangles and keeps the mesh closed
  for (let i = 0; i < loops.length; i++) {
    if (Math.abs(areas[i]) < 1e-18) fanLoop(xyz, tris, loops[i], n, capDir);
  }
  // outer loops (even depth) with their direct holes (odd depth, parent = outer)
  for (let i = 0; i < loops.length; i++) {
    if (Math.abs(areas[i]) < 1e-18 || depth[i] % 2 !== 0) continue;
    const ring: number[] = [];
    const ids: number[] = [];
    const holeIdx: number[] = [];
    for (const p of loops2d[i]) ring.push(p[0], p[1]);
    ids.push(...loops[i]);
    for (let j = 0; j < loops.length; j++) {
      if (parent[j] === i && depth[j] % 2 === 1 && Math.abs(areas[j]) >= 1e-18) {
        holeIdx.push(ids.length);
        for (const p of loops2d[j]) ring.push(p[0], p[1]);
        ids.push(...loops[j]);
      }
    }
    const tri = earcut(ring, holeIdx.length ? holeIdx : undefined, 2);
    if (dbg) dbg.loops[i].tris = tri.length / 3;
    // A valid triangulation of a polygon with h holes has n + 2h - 2 triangles.
    // earcut silently drops pieces of degenerate (self-touching, spiked)
    // rings, which would leave open edges; fall back to centroid fans, which
    // twin every boundary edge by construction and keep the signed cap area
    // (hence the volume) exact.
    const expected = ids.length + 2 * holeIdx.length - 2;
    if (tri.length / 3 !== expected) {
      const group = [i, ...loops.map((_, j) => j).filter((j) => parent[j] === i && depth[j] % 2 === 1 && Math.abs(areas[j]) >= 1e-18)];
      for (const j of group) fanLoop(xyz, tris, loops[j], n, capDir);
      continue;
    }
    // earcut's triangles share one orientation; decide the flip once from the
    // largest triangle so degenerate (zero-area) ones stay edge-consistent
    let bestAbs = 0, bestDot = 0;
    for (let k = 0; k < tri.length; k += 3) {
      const nx = triNormalDot(xyz, ids[tri[k]], ids[tri[k + 1]], ids[tri[k + 2]], n);
      if (Math.abs(nx) > bestAbs) { bestAbs = Math.abs(nx); bestDot = nx; }
    }
    const flip = bestDot * capDir < 0;
    for (let k = 0; k < tri.length; k += 3) {
      const i0 = ids[tri[k]];
      let i1 = ids[tri[k + 1]], i2 = ids[tri[k + 2]];
      if (flip) { const t = i1; i1 = i2; i2 = t; }
      tris.push(i0, i1, i2);
    }
  }
  return { ok: true };
}

/** Fan a boundary loop from its centroid; cap triangle (c, q, p) twins each loop edge p->q. */
function fanLoop(xyz: number[], tris: number[], loop: number[], n: [number, number, number], capDir: number) {
  let cx = 0, cy = 0, cz = 0;
  for (const v of loop) { cx += xyz[3 * v]; cy += xyz[3 * v + 1]; cz += xyz[3 * v + 2]; }
  const c = xyz.length / 3;
  xyz.push(cx / loop.length, cy / loop.length, cz / loop.length);
  void n; void capDir; // orientation follows the loop direction (edges are reversed)
  for (let k = 0; k < loop.length; k++) {
    const p = loop[k], q = loop[(k + 1) % loop.length];
    tris.push(c, q, p);
  }
}

function triNormalDot(xyz: number[], i0: number, i1: number, i2: number, n: [number, number, number]): number {
  const ax = xyz[3 * i1] - xyz[3 * i0], ay = xyz[3 * i1 + 1] - xyz[3 * i0 + 1], az = xyz[3 * i1 + 2] - xyz[3 * i0 + 2];
  const bx = xyz[3 * i2] - xyz[3 * i0], by = xyz[3 * i2 + 1] - xyz[3 * i0 + 1], bz = xyz[3 * i2 + 2] - xyz[3 * i0 + 2];
  const cx = ay * bz - az * by, cy = az * bx - ax * bz, cz = ax * by - ay * bx;
  return cx * n[0] + cy * n[1] + cz * n[2];
}

function cross(a: [number, number, number], b: [number, number, number]): [number, number, number] {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function normalizeInPlace(a: [number, number, number]) {
  const l = Math.hypot(a[0], a[1], a[2]) || 1;
  a[0] /= l; a[1] /= l; a[2] /= l;
}
function signedArea(p: [number, number][]): number {
  let s = 0;
  for (let i = 0; i < p.length; i++) {
    const j = (i + 1) % p.length;
    s += p[i][0] * p[j][1] - p[j][0] * p[i][1];
  }
  return s / 2;
}
function centroid2d(p: [number, number][], area: number): [number, number] {
  if (Math.abs(area) < 1e-18) return p[0];
  let cx = 0, cy = 0;
  for (let i = 0; i < p.length; i++) {
    const j = (i + 1) % p.length;
    const w = p[i][0] * p[j][1] - p[j][0] * p[i][1];
    cx += (p[i][0] + p[j][0]) * w;
    cy += (p[i][1] + p[j][1]) * w;
  }
  return [cx / (6 * area), cy / (6 * area)];
}
function pointInPolygon(pt: [number, number], poly: [number, number][]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i][0], yi = poly[i][1], xj = poly[j][0], yj = poly[j][1];
    if ((yi > pt[1]) !== (yj > pt[1]) && pt[0] < (xj - xi) * (pt[1] - yi) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

/** Drop unreferenced vertices and build a Mesh. */
function compact(xyz: number[], tris: number[]): Mesh {
  const remap = new Int32Array(xyz.length / 3).fill(-1);
  const out = new Uint32Array(tris.length);
  let nvOut = 0;
  for (let i = 0; i < tris.length; i++) {
    const v = tris[i];
    if (remap[v] < 0) remap[v] = nvOut++;
    out[i] = remap[v];
  }
  const verts = new Float32Array(3 * nvOut);
  for (let v = 0; v < remap.length; v++) {
    const r = remap[v];
    if (r < 0) continue;
    verts[3 * r] = xyz[3 * v]; verts[3 * r + 1] = xyz[3 * v + 1]; verts[3 * r + 2] = xyz[3 * v + 2];
  }
  return { verts, tris: out };
}
