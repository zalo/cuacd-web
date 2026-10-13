// The demo's worker protocol, exercised with Deno's Worker + WebGPU (the
// worker imports the browser bundle, which is rebuilt here first).
import { assert, assertAlmostEquals } from "@std/assert";
import { isClosed, unitCube, volume } from "../src/mesh.ts";
import { stairs } from "./fixtures.ts";

const demoDir = new URL("../demo/", import.meta.url);

async function bundle() {
  const cmd = new Deno.Command(Deno.execPath(), {
    args: ["bundle", "--platform", "browser", "--output", new URL("cuacd.bundle.js", demoDir).pathname, new URL("../src/mod.ts", import.meta.url).pathname],
    stdout: "null", stderr: "null",
  });
  const { success } = await cmd.output();
  assert(success, "deno bundle failed");
}

interface Msg { type: string; id?: number; [k: string]: unknown }

function startWorker(): { worker: Worker; next: (pred: (m: Msg) => boolean) => Promise<Msg>; all: Msg[] } {
  const worker = new Worker(new URL("worker.js", demoDir).href, { type: "module" });
  const all: Msg[] = [];
  const waiters: { pred: (m: Msg) => boolean; resolve: (m: Msg) => void }[] = [];
  worker.onmessage = (e) => {
    const m = e.data as Msg;
    all.push(m);
    for (let i = waiters.length - 1; i >= 0; i--) if (waiters[i].pred(m)) { waiters[i].resolve(m); waiters.splice(i, 1); }
  };
  const next = (pred: (m: Msg) => boolean) => new Promise<Msg>((resolve) => waiters.push({ pred, resolve }));
  return { worker, next, all };
}

Deno.test("demo worker: init, decompose with granular progress, cancel", async () => {
  await bundle();
  const { worker, next, all } = startWorker();
  try {
    worker.postMessage({ type: "init" });
    const ready = await next((m) => m.type === "ready" || m.type === "error");
    assert(ready.type === "ready", `worker init: ${ready.message}`);

    // a full run on the staircase
    const m = stairs();
    worker.postMessage({ type: "decompose", id: 1, verts: m.verts, tris: m.tris, opts: { threshold: 0.05, width: 30, nConcaveEdges: 16, maxIters: 50 } });
    const res = await next((x) => x.id === 1 && (x.type === "result" || x.type === "error"));
    assert(res.type === "result", `decompose: ${res.message}`);
    const parts = res.parts as { mesh: { verts: Float32Array; tris: Uint32Array }; hull: { verts: Float32Array; tris: Uint32Array }; meshVol: number }[];
    assert(parts.length >= 3 && parts.length <= 6, `parts=${parts.length}`);
    let total = 0;
    for (const p of parts) { assert(isClosed(p.mesh)); assert(isClosed(p.hull)); total += volume(p.mesh); }
    assertAlmostEquals(total, 6, 1e-3);
    // progress: monotone fractions, several phases, ends at 1
    const prog = all.filter((x) => x.type === "progress" && x.id === 1) as (Msg & { fraction: number; phase: string })[];
    assert(prog.length >= 5, `progress messages: ${prog.length}`);
    for (let i = 1; i < prog.length; i++) assert(prog[i].fraction >= prog[i - 1].fraction, "progress went backwards");
    const phases = new Set(prog.map((p) => p.phase));
    for (const ph of ["start", "select", "tree", "cut", "hull", "iteration", "done"]) assert(phases.has(ph), `missing phase ${ph}`);
    assert(prog[prog.length - 1].fraction === 1);

    // cancel: a long run on a cube with an unreachable threshold is aborted at an iteration boundary
    const c = unitCube();
    worker.postMessage({ type: "decompose", id: 2, verts: c.verts, tris: c.tris, opts: { threshold: 1e-9, maxIters: 200, mergeHulls: false } });
    await next((x) => x.type === "progress" && x.id === 2 && x.phase === "tree");
    worker.postMessage({ type: "cancel" });
    const end = await next((x) => x.id === 2 && (x.type === "result" || x.type === "error"));
    assert(end.type === "error" && end.aborted === true, `expected an aborted error, got ${end.type}`);

    // the worker is still usable afterwards
    worker.postMessage({ type: "decompose", id: 3, verts: c.verts, tris: c.tris, opts: { threshold: 0.05, maxIters: 5 } });
    const again = await next((x) => x.id === 3 && (x.type === "result" || x.type === "error"));
    assert(again.type === "result" && (again.parts as unknown[]).length === 1);
  } finally {
    worker.terminate();
  }
});
