// edit.js — interactive mesh modifications with manifold-3d.
//
// Runtime-agnostic (browser and Deno): every function takes the initialised
// manifold module (`await Module(); wasm.setup()`). Meshes are the port's
// plain { verts: Float32Array, tris: Uint32Array } records; Manifold objects
// live in wasm memory and are deleted here as soon as they are consumed.

/** Plain mesh → Manifold. Welds coincident vertices first; throws when the result is not a solid. */
export function toManifold(wasm, mesh) {
  const m = new wasm.Mesh({ numProp: 3, vertProperties: mesh.verts, triVerts: mesh.tris });
  m.merge();
  const man = wasm.Manifold.ofMesh(m);
  const st = man.status();
  if (st !== "NoError") {
    man.delete();
    throw new Error(`mesh is not a manifold solid (${st})`);
  }
  return man;
}

/** Manifold → plain mesh (positions only). Does not delete the manifold. */
export function fromManifold(man) {
  const m = man.getMesh();
  const np = m.numProp;
  let verts = m.vertProperties;
  if (np !== 3) {
    const n = verts.length / np;
    const out = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) { out[3 * i] = verts[np * i]; out[3 * i + 1] = verts[np * i + 1]; out[3 * i + 2] = verts[np * i + 2]; }
    verts = out;
  }
  return { verts: Float32Array.from(verts), tris: Uint32Array.from(m.triVerts) };
}

/** A primitive of the given shape ("sphere" | "box" | "cylinder") with extent `size`, centred at `center`. */
export function primitive(wasm, shape, center, size) {
  const M = wasm.Manifold;
  let p;
  if (shape === "sphere") p = M.sphere(size / 2, 32);
  else if (shape === "cylinder") p = M.cylinder(size, size / 2, size / 2, 32, true);
  else p = M.cube([size, size, size], true);
  const t = p.translate(center[0], center[1], center[2]);
  p.delete();
  return t;
}

/**
 * Boolean of the mesh with a primitive: op = "add" | "subtract" | "intersect".
 * Returns the new plain mesh (the input is untouched).
 */
export function booleanWithPrimitive(wasm, mesh, op, shape, center, size) {
  const a = toManifold(wasm, mesh);
  const b = primitive(wasm, shape, center, size);
  const r = op === "add" ? a.add(b) : op === "intersect" ? a.intersect(b) : a.subtract(b);
  a.delete(); b.delete();
  if (r.isEmpty()) { r.delete(); throw new Error("the operation left nothing"); }
  const out = fromManifold(r);
  r.delete();
  return out;
}

/** Twist about an axis (0 = x, 1 = y, 2 = z) by `degrees` across the mesh's extent along that axis. */
export function twist(wasm, mesh, degrees, axis = 1) {
  const a = toManifold(wasm, mesh);
  const box = a.boundingBox();
  const lo = box.min[axis], hi = box.max[axis];
  const span = Math.max(hi - lo, 1e-9);
  const cx = (box.min[(axis + 1) % 3] + box.max[(axis + 1) % 3]) / 2;
  const cy = (box.min[(axis + 2) % 3] + box.max[(axis + 2) % 3]) / 2;
  const rad = (degrees * Math.PI) / 180;
  const u = (axis + 1) % 3, w = (axis + 2) % 3;
  const r = a.warp((v) => {
    const t = (v[axis] - lo) / span;
    const ang = rad * t;
    const x = v[u] - cx, y = v[w] - cy;
    const c = Math.cos(ang), s = Math.sin(ang);
    v[u] = cx + c * x - s * y;
    v[w] = cy + s * x + c * y;
  });
  a.delete();
  const out = fromManifold(r);
  r.delete();
  return out;
}

/** Convex hull of the mesh. */
export function hullOf(wasm, mesh) {
  const a = toManifold(wasm, mesh);
  const h = a.hull();
  a.delete();
  const out = fromManifold(h);
  h.delete();
  return out;
}

/** Signed volume and manifold status of a plain mesh, for display. */
export function inspect(wasm, mesh) {
  try {
    const a = toManifold(wasm, mesh);
    const r = { volume: a.volume(), area: a.surfaceArea(), status: "NoError", genus: a.genus() };
    a.delete();
    return r;
  } catch (e) {
    return { volume: NaN, area: NaN, status: e.message, genus: NaN };
  }
}
