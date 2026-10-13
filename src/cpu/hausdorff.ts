// hausdorff.ts — bidirectional sampled Hausdorff distance between two meshes
// (port of cuda/hausdorff.cuh: area-proportional sampling with Wang-hash
// barycentrics, nearest triangle through a small BVH).

import type { Mesh } from "../mesh.ts";

export const HD_RESOLUTION = 2000;
export const HD_MIN_SAMPLES = 1000;

function wangHash(seed: number): number {
  seed = ((seed ^ 61) ^ (seed >>> 16)) >>> 0;
  seed = Math.imul(seed, 9) >>> 0;
  seed = (seed ^ (seed >>> 4)) >>> 0;
  seed = Math.imul(seed, 0x27d4eb2d) >>> 0;
  seed = (seed ^ (seed >>> 15)) >>> 0;
  return seed;
}

function triArea(V: Float32Array, a: number, b: number, c: number): number {
  const abx = V[b] - V[a], aby = V[b + 1] - V[a + 1], abz = V[b + 2] - V[a + 2];
  const acx = V[c] - V[a], acy = V[c + 1] - V[a + 1], acz = V[c + 2] - V[a + 2];
  const nx = aby * acz - abz * acy, ny = abz * acx - abx * acz, nz = abx * acy - aby * acx;
  return 0.5 * Math.sqrt(nx * nx + ny * ny + nz * nz);
}

/** Area-proportional surface samples (xyz interleaved). */
export function sampleMesh(m: Mesh): Float32Array {
  const V = m.verts, T = m.tris;
  const nt = T.length / 3;
  let area = 0;
  const areas = new Float32Array(nt);
  for (let t = 0; t < nt; t++) {
    areas[t] = triArea(V, T[3 * t] * 3, T[3 * t + 1] * 3, T[3 * t + 2] * 3);
    area += areas[t];
  }
  let res = Math.floor(HD_RESOLUTION * area);
  if (res < HD_MIN_SAMPLES) res = HD_MIN_SAMPLES;
  const counts = new Int32Array(nt);
  let total = 0;
  const step = nt > res ? Math.floor(nt / res) : 0;
  for (let t = 0; t < nt; t++) {
    const areaCount = area > 1e-20 ? Math.floor(res / area * areas[t]) : 0;
    let N: number;
    if (nt > res) N = areaCount > 0 ? areaCount : (t % step === 0 ? 1 : 0);
    else N = areaCount > 0 ? areaCount : (t % 2 === 0 ? 1 : 0);
    counts[t] = N;
    total += N;
  }
  const out = new Float32Array(total * 3);
  let idx = 0;
  for (let t = 0; t < nt; t++) {
    const a = T[3 * t] * 3, b = T[3 * t + 1] * 3, c = T[3 * t + 2] * 3;
    for (let k = 0; k < counts[t]; k++) {
      const ha = wangHash((Math.imul(t, 65537) + k) >>> 0);
      const hb = wangHash((ha + 0x9e3779b9) >>> 0);
      const ra = (ha & 0xffffff) / 0xffffff;
      const rb = (hb & 0xffffff) / 0xffffff;
      const sqa = Math.sqrt(ra);
      const w0 = 1 - sqa, w1 = sqa * (1 - rb), w2 = sqa * rb;
      out[3 * idx] = w0 * V[a] + w1 * V[b] + w2 * V[c];
      out[3 * idx + 1] = w0 * V[a + 1] + w1 * V[b + 1] + w2 * V[c + 1];
      out[3 * idx + 2] = w0 * V[a + 2] + w1 * V[b + 2] + w2 * V[c + 2];
      idx++;
    }
  }
  return out;
}

function distPtSegSq(px: number, py: number, pz: number, ax: number, ay: number, az: number, bx: number, by: number, bz: number): number {
  const abx = bx - ax, aby = by - ay, abz = bz - az;
  const apx = px - ax, apy = py - ay, apz = pz - az;
  const ab2 = abx * abx + aby * aby + abz * abz;
  let t = ab2 > 1e-20 ? (apx * abx + apy * aby + apz * abz) / ab2 : 0;
  t = Math.max(0, Math.min(1, t));
  const dx = apx - t * abx, dy = apy - t * aby, dz = apz - t * abz;
  return dx * dx + dy * dy + dz * dz;
}

export function distPtTriSq(
  px: number, py: number, pz: number,
  ax: number, ay: number, az: number,
  bx: number, by: number, bz: number,
  cx: number, cy: number, cz: number,
): number {
  const abx = bx - ax, aby = by - ay, abz = bz - az;
  const acx = cx - ax, acy = cy - ay, acz = cz - az;
  const apx = px - ax, apy = py - ay, apz = pz - az;
  const nx = aby * acz - abz * acy, ny = abz * acx - abx * acz, nz = abx * acy - aby * acx;
  const n2 = nx * nx + ny * ny + nz * nz;
  if (n2 < 1e-20) {
    const da = apx * apx + apy * apy + apz * apz;
    const bpx = px - bx, bpy = py - by, bpz = pz - bz;
    const db = bpx * bpx + bpy * bpy + bpz * bpz;
    const cpx = px - cx, cpy = py - cy, cpz = pz - cz;
    const dc = cpx * cpx + cpy * cpy + cpz * cpz;
    return Math.min(da, db, dc);
  }
  const d00 = abx * abx + aby * aby + abz * abz;
  const d01 = abx * acx + aby * acy + abz * acz;
  const d02 = abx * apx + aby * apy + abz * apz;
  const d11 = acx * acx + acy * acy + acz * acz;
  const d12 = acx * apx + acy * apy + acz * apz;
  const inv = 1 / (d00 * d11 - d01 * d01);
  const u = (d11 * d02 - d01 * d12) * inv;
  const v = (d00 * d12 - d01 * d02) * inv;
  if (u >= 0 && v >= 0 && u + v <= 1) {
    const dist = nx * apx + ny * apy + nz * apz;
    return (dist * dist) / n2;
  }
  return Math.min(
    distPtSegSq(px, py, pz, ax, ay, az, bx, by, bz),
    distPtSegSq(px, py, pz, bx, by, bz, cx, cy, cz),
    distPtSegSq(px, py, pz, cx, cy, cz, ax, ay, az),
  );
}

/** Flat binary BVH over triangles (median split along the longest axis). */
export class TriBVH {
  readonly V: Float32Array;
  readonly T: Uint32Array;
  // node arrays
  readonly bmin: Float32Array;
  readonly bmax: Float32Array;
  readonly left: Int32Array; // child or -1
  readonly right: Int32Array;
  readonly start: Int32Array; // leaf: range into order
  readonly count: Int32Array;
  readonly order: Uint32Array;
  nNodes = 0;

  constructor(m: Mesh, leafSize = 4) {
    this.V = m.verts; this.T = m.tris;
    const nt = m.tris.length / 3;
    const cap = Math.max(1, 2 * Math.ceil(nt / Math.max(1, Math.floor(leafSize / 2))) + 2);
    this.bmin = new Float32Array(cap * 3); this.bmax = new Float32Array(cap * 3);
    this.left = new Int32Array(cap).fill(-1); this.right = new Int32Array(cap).fill(-1);
    this.start = new Int32Array(cap); this.count = new Int32Array(cap);
    this.order = new Uint32Array(nt);
    for (let i = 0; i < nt; i++) this.order[i] = i;
    const cent = new Float32Array(nt * 3);
    const V = m.verts, T = m.tris;
    for (let t = 0; t < nt; t++) {
      for (let a = 0; a < 3; a++) cent[3 * t + a] = (V[T[3 * t] * 3 + a] + V[T[3 * t + 1] * 3 + a] + V[T[3 * t + 2] * 3 + a]) / 3;
    }
    if (nt > 0) this.build(0, nt, cent, leafSize);
  }

  private build(s: number, e: number, cent: Float32Array, leafSize: number): number {
    const node = this.nNodes++;
    const V = this.V, T = this.T, order = this.order;
    let x0 = Infinity, y0 = Infinity, z0 = Infinity, x1 = -Infinity, y1 = -Infinity, z1 = -Infinity;
    let cx0 = Infinity, cy0 = Infinity, cz0 = Infinity, cx1 = -Infinity, cy1 = -Infinity, cz1 = -Infinity;
    for (let i = s; i < e; i++) {
      const t = order[i];
      for (let k = 0; k < 3; k++) {
        const v = T[3 * t + k] * 3;
        const x = V[v], y = V[v + 1], z = V[v + 2];
        if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; if (z < z0) z0 = z; if (z > z1) z1 = z;
      }
      const cx = cent[3 * t], cy = cent[3 * t + 1], cz = cent[3 * t + 2];
      if (cx < cx0) cx0 = cx; if (cx > cx1) cx1 = cx; if (cy < cy0) cy0 = cy; if (cy > cy1) cy1 = cy; if (cz < cz0) cz0 = cz; if (cz > cz1) cz1 = cz;
    }
    this.bmin[3 * node] = x0; this.bmin[3 * node + 1] = y0; this.bmin[3 * node + 2] = z0;
    this.bmax[3 * node] = x1; this.bmax[3 * node + 1] = y1; this.bmax[3 * node + 2] = z1;
    if (e - s <= leafSize) {
      this.start[node] = s; this.count[node] = e - s;
      return node;
    }
    const ex = cx1 - cx0, ey = cy1 - cy0, ez = cz1 - cz0;
    const axis = ex >= ey && ex >= ez ? 0 : ey >= ez ? 1 : 2;
    const sub = order.subarray(s, e);
    sub.sort((p, q) => cent[3 * p + axis] - cent[3 * q + axis]);
    const mid = (s + e) >> 1;
    this.count[node] = 0;
    this.left[node] = this.build(s, mid, cent, leafSize);
    this.right[node] = this.build(mid, e, cent, leafSize);
    return node;
  }

  private boxDistSq(node: number, px: number, py: number, pz: number): number {
    const dx = Math.max(this.bmin[3 * node] - px, 0, px - this.bmax[3 * node]);
    const dy = Math.max(this.bmin[3 * node + 1] - py, 0, py - this.bmax[3 * node + 1]);
    const dz = Math.max(this.bmin[3 * node + 2] - pz, 0, pz - this.bmax[3 * node + 2]);
    return dx * dx + dy * dy + dz * dz;
  }

  /** Squared distance from a point to the nearest triangle. */
  nearestSq(px: number, py: number, pz: number, bestIn = Infinity): number {
    if (this.nNodes === 0) return Infinity;
    let best = bestIn;
    const stack = new Int32Array(64);
    let sp = 0;
    stack[sp++] = 0;
    const V = this.V, T = this.T;
    while (sp > 0) {
      const node = stack[--sp];
      if (this.boxDistSq(node, px, py, pz) >= best) continue;
      if (this.count[node] > 0) {
        const s = this.start[node], e = s + this.count[node];
        for (let i = s; i < e; i++) {
          const t = this.order[i];
          const a = T[3 * t] * 3, b = T[3 * t + 1] * 3, c = T[3 * t + 2] * 3;
          const d = distPtTriSq(px, py, pz, V[a], V[a + 1], V[a + 2], V[b], V[b + 1], V[b + 2], V[c], V[c + 1], V[c + 2]);
          if (d < best) best = d;
        }
      } else {
        const l = this.left[node], r = this.right[node];
        const dl = this.boxDistSq(l, px, py, pz), dr = this.boxDistSq(r, px, py, pz);
        if (dl < dr) { stack[sp++] = r; stack[sp++] = l; } else { stack[sp++] = l; stack[sp++] = r; }
      }
    }
    return best;
  }
}

/** max over samples of a of the distance to mesh b (one direction). */
export function directedDistance(samplesA: Float32Array, bvhB: TriBVH): number {
  let worst = 0;
  for (let i = 0; i < samplesA.length; i += 3) {
    const d = bvhB.nearestSq(samplesA[i], samplesA[i + 1], samplesA[i + 2]);
    if (d > worst) worst = d;
  }
  return Math.sqrt(worst);
}

export function hausdorff(a: Mesh, b: Mesh): number {
  if (a.tris.length === 0 || b.tris.length === 0) return 0;
  const sa = sampleMesh(a), sb = sampleMesh(b);
  const ba = new TriBVH(a), bb = new TriBVH(b);
  return Math.max(directedDistance(sa, bb), directedDistance(sb, ba));
}
