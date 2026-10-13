// worker.js — runs the convex decomposition off the main thread.
//
// Protocol (postMessage):
//   → { type: "init" }                               ← { type: "ready", info } | { type: "error", message }
//   → { type: "decompose", id, verts, tris, opts }   ← { type: "progress", id, phase, fraction, message, ... } (many)
//                                                    ← { type: "result", id, parts: [{ mesh, hull, meshVol, hullVol, hausdorff, merged }], stats, iterations }
//                                                    ← { type: "error", id, message, aborted }
//   → { type: "cancel" }                             (aborts the running decomposition at its next boundary)
//
// WebGPU is acquired inside the worker (dedicated workers have navigator.gpu);
// the main thread only renders. Buffers of the result are transferred, not
// copied.

import { Decomposer, getDevice } from "./cuacd.bundle.js";

let device = null;
let info = null;
let running = null; // AbortController of the running job

const post = (msg, transfer) => self.postMessage(msg, transfer ?? []);

self.onmessage = async (e) => {
  const msg = e.data;
  if (msg.type === "init") {
    try {
      ({ device, info } = await getDevice());
      post({ type: "ready", info });
    } catch (err) {
      post({ type: "error", message: `WebGPU is not available in this worker: ${err.message}` });
    }
    return;
  }
  if (msg.type === "cancel") {
    running?.abort();
    return;
  }
  if (msg.type !== "decompose") return;
  if (!device) { post({ type: "error", id: msg.id, message: "worker not initialised" }); return; }
  if (running) running.abort();
  const ctrl = new AbortController();
  running = ctrl;
  const d = new Decomposer(device, {
    ...msg.opts,
    signal: ctrl.signal,
    onProgress: (p) => post({ type: "progress", id: msg.id, ...p }),
  });
  const t0 = performance.now();
  try {
    const r = await d.decompose({ verts: msg.verts, tris: msg.tris });
    const transfer = [];
    const parts = r.parts.map((p) => {
      transfer.push(p.mesh.verts.buffer, p.mesh.tris.buffer, p.hull.verts.buffer, p.hull.tris.buffer);
      return {
        mesh: { verts: p.mesh.verts, tris: p.mesh.tris },
        hull: { verts: p.hull.verts, tris: p.hull.tris },
        meshVol: p.meshVol, hullVol: p.hullVol, hausdorff: p.hausdorff, merged: p.merged ?? 1, stuck: p.stuck,
      };
    });
    post({
      type: "result", id: msg.id, parts, iterations: r.iterations,
      stats: { ...r.stats, wallMs: performance.now() - t0 }, merge: r.merge,
    }, transfer);
  } catch (err) {
    post({ type: "error", id: msg.id, message: err.message, aborted: err.name === "AbortError" });
  } finally {
    d.destroy();
    if (running === ctrl) running = null;
  }
};
