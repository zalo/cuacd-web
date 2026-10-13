// The demo's manifold-3d editing helpers, run against the npm package in Deno.
import { assert, assertAlmostEquals } from "@std/assert";
import Module from "manifold-3d";
import { isClosed, unitCube, volume } from "../src/mesh.ts";
import { booleanWithPrimitive, fromManifold, hullOf, inspect, toManifold, twist } from "../demo/edit.js";
import { stairs } from "./fixtures.ts";

const wasm = await Module();
wasm.setup();

Deno.test("edit: plain mesh round trip through manifold", () => {
  const m = toManifold(wasm, stairs());
  assertAlmostEquals(m.volume(), 6, 1e-6);
  const back = fromManifold(m);
  m.delete();
  assert(isClosed(back));
  assertAlmostEquals(volume(back), 6, 1e-6);
});

Deno.test("edit: subtract and add a sphere at a point", () => {
  const cube = unitCube();
  const carved = booleanWithPrimitive(wasm, cube, "subtract", "sphere", [0.5, 0.5, 1], 0.5);
  assert(isClosed(carved));
  assert(volume(carved) < 1 && volume(carved) > 1 - (4 / 3) * Math.PI * 0.125 / 2 - 1e-6, `volume ${volume(carved)}`);
  const bumped = booleanWithPrimitive(wasm, cube, "add", "box", [1, 0.5, 0.5], 0.4);
  assert(isClosed(bumped));
  assertAlmostEquals(volume(bumped), 1 + 0.4 * 0.4 * 0.2, 1e-5);
  const cyl = booleanWithPrimitive(wasm, cube, "intersect", "cylinder", [0.5, 0.5, 0.5], 1);
  assert(isClosed(cyl) && volume(cyl) < 0.8 && volume(cyl) > 0.7, `cylinder volume ${volume(cyl)}`); // pi/4 for the inscribed cylinder
});

Deno.test("edit: twist keeps the mesh closed and its volume", () => {
  const t = twist(wasm, stairs(), 60, 1);
  assert(isClosed(t));
  assertAlmostEquals(volume(t), 6, 0.3);
  const info = inspect(wasm, t);
  assert(info.status === "NoError" && info.genus === 0);
});

Deno.test("edit: hull of a staircase cuts the stepped corner", () => {
  const h = hullOf(wasm, stairs());
  assert(isClosed(h));
  assertAlmostEquals(volume(h), 7, 1e-5); // 3x3x1 prism minus the triangle between (3,1) and (1,3)
});

Deno.test("edit: a non-manifold mesh is rejected with a message", () => {
  const m = unitCube();
  const broken = { verts: m.verts, tris: m.tris.slice(0, m.tris.length - 3) };
  let msg = "";
  try { toManifold(wasm, broken); } catch (e) { msg = (e as Error).message; }
  assert(/manifold/i.test(msg), msg);
});
