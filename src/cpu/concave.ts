// concave.ts — candidate cutting planes from concave mesh edges (port of
// cuda/la_concave.cu). For each sampled concave edge: two face-offset planes
// and two bisector planes (the latter skipped for nearly opposite normals).

import { edgeCSR, type Mesh } from "../mesh.ts";

export interface ConcaveOptions {
  nConcaveEdges: number; // reservoir size
  concaveEps: number; // plane offset
  concaveThreshold: number; // radians, dihedral angle above which an edge is concave
  seed?: number;
}

function faceNormal(V: Float32Array, T: Uint32Array, f: number): [number, number, number] | null {
  const a = T[3 * f] * 3, b = T[3 * f + 1] * 3, c = T[3 * f + 2] * 3;
  const e1x = V[b] - V[a], e1y = V[b + 1] - V[a + 1], e1z = V[b + 2] - V[a + 2];
  const e2x = V[c] - V[a], e2y = V[c + 1] - V[a + 1], e2z = V[c + 2] - V[a + 2];
  const nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
  const l = Math.sqrt(nx * nx + ny * ny + nz * nz);
  if (l < 1e-12) return null;
  return [nx / l, ny / l, nz / l];
}

/** Returns planes as a flat Float32Array (4 floats per plane). */
export function concaveEdgePlanes(m: Mesh, o: ConcaveOptions): Float32Array {
  const V = m.verts, T = m.tris;
  const nt = T.length / 3;
  if (o.nConcaveEdges <= 0 || nt === 0) return new Float32Array(0);
  // undirected edges in (min, max) order with their two faces (lower face
  // index first), from the directed-edge CSR
  const { off, dst, face } = edgeCSR(m);
  const nvv = off.length - 1;
  const findFace = (a: number, b: number) => {
    for (let k = off[a]; k < off[a + 1]; k++) if (dst[k] === b) return face[k];
    return -1;
  };
  // deterministic reservoir sampling in sorted edge order
  let rng = (o.seed ?? 12345) >>> 0;
  const lcg = () => { rng = (Math.imul(rng, 1664525) + 1013904223) >>> 0; return rng; };
  const reservoir: [number, number, number, number][] = [];
  let seen = 0;
  const partners: number[] = [];
  for (let v0 = 0; v0 < nvv; v0++) {
    partners.length = 0;
    for (let k = off[v0]; k < off[v0 + 1]; k++) { const b = dst[k]; if (b > v0 && !partners.includes(b)) partners.push(b); }
    partners.sort((x, y) => x - y);
    for (const v1 of partners) {
      const fa = findFace(v0, v1), fb = findFace(v1, v0);
      if (fa < 0 || fb < 0) continue;
      const f0 = Math.min(fa, fb), f1 = Math.max(fa, fb);
      const n0 = faceNormal(V, T, f0), n1 = faceNormal(V, T, f1);
      if (!n0 || !n1) continue;
      let cosA = n0[0] * n1[0] + n0[1] * n1[1] + n0[2] * n1[2];
      cosA = Math.max(-1, Math.min(1, cosA));
      const alpha = Math.acos(cosA);
      let vOther = -1;
      for (let i = 0; i < 3; i++) { const vv = T[3 * f0 + i]; if (vv !== v0 && vv !== v1) { vOther = vv; break; } }
      if (vOther < 0) continue;
      const dx = V[3 * vOther] - V[3 * v0], dy = V[3 * vOther + 1] - V[3 * v0 + 1], dz = V[3 * vOther + 2] - V[3 * v0 + 2];
      const d = dx * n1[0] + dy * n1[1] + dz * n1[2];
      const dihedral = d > 0 ? Math.PI + alpha : Math.PI - alpha;
      if (!(dihedral > o.concaveThreshold)) continue;
      seen++;
      if (reservoir.length < o.nConcaveEdges) reservoir.push([v0, v1, f0, f1]);
      else {
        const r = lcg();
        const thresh = Math.floor(o.nConcaveEdges / seen * 4294967295);
        if (r < thresh) {
          const slot = lcg() % o.nConcaveEdges;
          reservoir[slot] = [v0, v1, f0, f1];
        }
      }
    }
  }
  const out: number[] = [];
  for (const [v0, v1, f0, f1] of reservoir) {
    const n0 = faceNormal(V, T, f0), n1 = faceNormal(V, T, f1);
    if (!n0 || !n1) continue;
    const mx = (V[3 * v0] + V[3 * v1]) / 2, my = (V[3 * v0 + 1] + V[3 * v1 + 1]) / 2, mz = (V[3 * v0 + 2] + V[3 * v1 + 2]) / 2;
    const eps = o.concaveEps;
    out.push(n1[0], n1[1], n1[2], -(n1[0] * mx + n1[1] * my + n1[2] * mz + eps));
    out.push(n0[0], n0[1], n0[2], -(n0[0] * mx + n0[1] * my + n0[2] * mz + eps));
    let bx = n0[0] + n1[0], by = n0[1] + n1[1], bz = n0[2] + n1[2];
    const bl = Math.sqrt(bx * bx + by * by + bz * bz);
    if (bl >= 1e-4) {
      bx /= bl; by /= bl; bz /= bl;
      out.push(bx, by, bz, -(bx * mx + by * my + bz * mz + eps));
      out.push(bx, by, bz, -(bx * mx + by * my + bz * mz - eps));
    }
  }
  return new Float32Array(out);
}
