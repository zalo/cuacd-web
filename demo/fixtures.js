// Small procedural sample for the demo (same staircase as tests/fixtures.ts).
export function stairs() {
  const poly = [[0, 0], [3, 0], [3, 1], [2, 1], [2, 2], [1, 2], [1, 3], [0, 3]];
  const n = poly.length, height = 1;
  const verts = [];
  for (const z of [0, height]) for (const [x, y] of poly) verts.push(x, y, z);
  const tris = [];
  const idx = poly.map((_, i) => i);
  const area2 = (a, b, c) => (poly[b][0] - poly[a][0]) * (poly[c][1] - poly[a][1]) - (poly[b][1] - poly[a][1]) * (poly[c][0] - poly[a][0]);
  const inside = (p, a, b, c) => area2(a, b, p) >= 0 && area2(b, c, p) >= 0 && area2(c, a, p) >= 0;
  let guard = 0;
  while (idx.length > 3 && guard++ < 1000) {
    let clipped = false;
    for (let i = 0; i < idx.length; i++) {
      const a = idx[(i + idx.length - 1) % idx.length], b = idx[i], c = idx[(i + 1) % idx.length];
      if (area2(a, b, c) <= 0) continue;
      if (idx.some((p) => p !== a && p !== b && p !== c && inside(p, a, b, c))) continue;
      tris.push(a, c, b, n + a, n + b, n + c);
      idx.splice(i, 1); clipped = true; break;
    }
    if (!clipped) break;
  }
  tris.push(idx[0], idx[2], idx[1], n + idx[0], n + idx[1], n + idx[2]);
  for (let i = 0; i < n; i++) { const j = (i + 1) % n; tris.push(i, j, n + j, i, n + j, n + i); }
  return { verts: new Float32Array(verts), tris: new Uint32Array(tris) };
}
