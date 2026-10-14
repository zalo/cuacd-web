// Exercise the built npm package from Node: the CPU path always, the GPU path
// when Dawn's `webgpu` npm package is available (DAWN_NODE_DIR, default /tmp/dawn_node).
//   node scripts/test_npm.mjs              # the package in this repository (dist/)
//   node scripts/test_npm.mjs cuacd-web    # an installed package, resolved from the cwd
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";
const spec = process.argv[2]
  ? createRequire(pathToFileURL(process.cwd() + "/").href).resolve(process.argv[2])
  : new URL("../dist/esm/src/mod.js", import.meta.url).pathname;
console.log(`testing ${spec}`);
const pkg = await import(pathToFileURL(spec).href);
const { planeCut, convexHull, initHull, isClosed, volume, concaveEdgePlanes, unitCube, lShape, decompose } = pkg;

// --- CPU path
const cube = unitCube();
const cut = planeCut(cube, [1, 0, 0, -0.5]);
if (!cut.ok || Math.abs(volume(cut.pos) - 0.5) > 1e-6 || !isClosed(cut.pos)) throw new Error("planeCut failed");
if (!(await initHull())) throw new Error("wasm hull failed to load");
const h = convexHull(lShape().verts);
if (Math.abs(h.volume - 1.75) > 1e-6 || !isClosed(h.mesh)) throw new Error(`hull failed: ${h.volume}`); // cube + half box + the wedge between them
const planes = concaveEdgePlanes(lShape(), { nConcaveEdges: 4, concaveEps: 0.005, concaveThreshold: 3.49 });
console.log(`CPU ok: cut ${cut.pos.tris.length / 3}+${cut.neg.tris.length / 3} tris, hull vol ${h.volume}, ${planes.length / 4} concave planes`);

// --- GPU path through Dawn
const dawnDir = process.env.DAWN_NODE_DIR ?? "/tmp/dawn_node";
let dawn;
try { dawn = createRequire(dawnDir + "/package.json")("webgpu"); } catch (e) { console.log(`GPU skipped: no Dawn bindings at ${dawnDir} (${e.message})`); process.exit(0); }
Object.assign(globalThis, dawn.globals);
Object.defineProperty(globalThis, "navigator", { value: { gpu: dawn.create([]) }, configurable: true });
const t0 = performance.now();
const r = await decompose(lShape(), { threshold: 0.05, maxIters: 20 });
console.log(`GPU ok (Dawn): L-shape → ${r.parts.length} parts in ${r.iterations} iterations, ${(performance.now() - t0).toFixed(0)} ms`);
if (r.parts.length < 2 || r.parts.length > 10) throw new Error("unexpected part count");
for (const p of r.parts) if (!isClosed(p.mesh) || !isClosed(p.hull)) throw new Error("part not closed");
process.exit(0);
