import type { Mesh } from "../src/mesh.ts";

/** Square torus: outer half-width 1, inner 0.5, height 0.5. Volume = (4-1)*0.5 = 1.5. */
export function squareTorus(): Mesh {
  const R = 1, r = 0.5, h = 0.25;
  const verts: number[] = [];
  const corner = (s: number, z: number) => [[-s, -s, z], [s, -s, z], [s, s, z], [-s, s, z]];
  // 0-3 outer bottom, 4-7 outer top, 8-11 inner bottom, 12-15 inner top
  for (const [s, z] of [[R, -h], [R, h], [r, -h], [r, h]] as [number, number][]) for (const c of corner(s, z)) verts.push(...c);
  const tris: number[] = [];
  const quad = (a: number, b: number, c: number, d: number) => tris.push(a, b, c, a, c, d);
  for (let i = 0; i < 4; i++) {
    const j = (i + 1) % 4;
    quad(i, j, 4 + j, 4 + i); // outer wall (outward)
    quad(8 + j, 8 + i, 12 + i, 12 + j); // inner wall (facing the hole)
    quad(4 + i, 4 + j, 12 + j, 12 + i); // top annulus (+z)
    quad(j, i, 8 + i, 8 + j); // bottom annulus (-z)
  }
  return { verts: new Float32Array(verts), tris: new Uint32Array(tris) };
}

/** Welded staircase prism (3 steps): needs at least 3 convex parts. Volume 6. */
export function stairs(): Mesh {
  return extrudePolygon([[0, 0], [3, 0], [3, 1], [2, 1], [2, 2], [1, 2], [1, 3], [0, 3]], 1);
}

/** Extrude a simple CCW polygon along +z into a closed, outward-wound prism. */
export function extrudePolygon(poly: number[][], height: number): Mesh {
  const n = poly.length;
  const verts: number[] = [];
  for (const z of [0, height]) for (const [x, y] of poly) verts.push(x, y, z);
  const tris: number[] = [];
  // ear-clip the polygon (simple polygon, CCW)
  const idx = poly.map((_, i) => i);
  const area2 = (a: number, b: number, c: number) =>
    (poly[b][0] - poly[a][0]) * (poly[c][1] - poly[a][1]) - (poly[b][1] - poly[a][1]) * (poly[c][0] - poly[a][0]);
  const inside = (p: number, a: number, b: number, c: number) =>
    area2(a, b, p) >= 0 && area2(b, c, p) >= 0 && area2(c, a, p) >= 0;
  let guard = 0;
  while (idx.length > 3 && guard++ < 1000) {
    let clipped = false;
    for (let i = 0; i < idx.length; i++) {
      const a = idx[(i + idx.length - 1) % idx.length], b = idx[i], c = idx[(i + 1) % idx.length];
      if (area2(a, b, c) <= 0) continue;
      let ok = true;
      for (const p of idx) if (p !== a && p !== b && p !== c && inside(p, a, b, c)) { ok = false; break; }
      if (!ok) continue;
      tris.push(a, c, b); // bottom (-z)
      tris.push(n + a, n + b, n + c); // top (+z)
      idx.splice(i, 1);
      clipped = true;
      break;
    }
    if (!clipped) break;
  }
  tris.push(idx[0], idx[2], idx[1], n + idx[0], n + idx[1], n + idx[2]);
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    tris.push(i, j, n + j, i, n + j, n + i);
  }
  return { verts: new Float32Array(verts), tris: new Uint32Array(tris) };
}

export async function loadSTL(path: string): Promise<Mesh> {
  const { parseSTL } = await import("../src/mesh.ts");
  const buf = await Deno.readFile(path);
  return parseSTL(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer);
}
export async function loadOBJ(path: string): Promise<Mesh> {
  const { parseOBJ } = await import("../src/mesh.ts");
  return parseOBJ(await Deno.readTextFile(path));
}

/** Welded L prism (L polygon in xy extruded along z); has one concave vertical edge. */
export function lPrism(): Mesh {
  return extrudePolygon([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], 1);
}
