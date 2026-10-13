// main.js — interactive demo: load a mesh, edit it with manifold-3d, decompose
// it on WebGPU in a worker, and show the convex hulls in colour.
//
// Structure (after zalo/StochasticLatticeGen): a World owns the three.js
// scene; the decomposition runs in ./worker.js and reports granular progress;
// a lil-gui panel holds the parameters; files can be dropped on the page.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GUI } from "three/addons/libs/lil-gui.module.min.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";
import Module from "manifold-3d";
import { isClosed, nt, nv, parseOBJ, parseSTL, weld } from "./cuacd.bundle.js";
import { stairs } from "./fixtures.js";
import { booleanWithPrimitive, hullOf, inspect, twist } from "./edit.js";

// ---------------------------------------------------------------- status bar
const status = {
  msg: document.getElementById("msg"),
  fill: document.getElementById("fill"),
  phase: document.getElementById("phase"),
  sub: document.getElementById("sub"),
  set(message, fraction = null, sub = "") {
    this.msg.textContent = message;
    if (fraction !== null) this.fill.style.width = `${(100 * fraction).toFixed(1)}%`;
    this.sub.textContent = sub;
  },
  ghost(fraction) { this.phase.style.width = `${(100 * fraction).toFixed(1)}%`; },
};
addEventListener("error", (e) => status.set(`error: ${e.message}`));
addEventListener("unhandledrejection", (e) => status.set(`error: ${e.reason?.message ?? e.reason}`));

// ---------------------------------------------------------------- world
class World {
  constructor() {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x1c1f26);
    this.camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.01, 1000);
    this.camera.position.set(3, 2, 4);
    // without WebGL (e.g. a headless browser) the app still loads models,
    // edits them and decomposes them; only the view is missing
    try {
      this.renderer = new THREE.WebGLRenderer({ antialias: true });
    } catch (e) {
      this.renderer = null;
      this.canvas = document.createElement("canvas");
      document.body.appendChild(this.canvas);
      setTimeout(() => status.set(`no WebGL context (${e.message}); rendering disabled`), 0);
    }
    const canvas = this.renderer ? this.renderer.domElement : this.canvas;
    if (this.renderer) {
      this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
      this.renderer.setSize(innerWidth, innerHeight);
      document.body.appendChild(canvas);
    }
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x3a4050, 1.1));
    const key = new THREE.DirectionalLight(0xffffff, 1.6); key.position.set(5, 8, 3); this.scene.add(key);
    const rim = new THREE.DirectionalLight(0x88aaff, 0.5); rim.position.set(-4, 2, -5); this.scene.add(rim);
    this.grid = new THREE.GridHelper(10, 20, 0x3a4050, 0x2a2f3a);
    this.scene.add(this.grid);
    this.inputGroup = new THREE.Group();
    this.partsGroup = new THREE.Group();
    this.toolGroup = new THREE.Group();
    this.scene.add(this.inputGroup, this.partsGroup, this.toolGroup);
    addEventListener("resize", () => {
      this.camera.aspect = innerWidth / innerHeight;
      this.camera.updateProjectionMatrix();
      this.renderer?.setSize(innerWidth, innerHeight);
    });
    this.renderer?.setAnimationLoop(() => { this.controls.update(); this.renderer.render(this.scene, this.camera); });
  }
  get domElement() { return this.renderer ? this.renderer.domElement : this.canvas; }
  frame(radius, center) {
    this.controls.target.copy(center);
    this.camera.position.copy(center).add(new THREE.Vector3(1, 0.7, 1.3).normalize().multiplyScalar(radius * 2.6));
    this.camera.near = radius / 100; this.camera.far = radius * 100; this.camera.updateProjectionMatrix();
    this.grid.position.set(center.x, center.y - radius, center.z);
    this.grid.scale.setScalar(radius / 2.5);
  }
}

function toGeometry(m) {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(m.verts, 3));
  g.setIndex(new THREE.BufferAttribute(m.tris, 1));
  g.computeVertexNormals();
  g.computeBoundingSphere();
  return g;
}
function centroid(m) {
  const c = new THREE.Vector3();
  for (let i = 0; i < m.verts.length; i += 3) c.x += m.verts[i], c.y += m.verts[i + 1], c.z += m.verts[i + 2];
  return c.multiplyScalar(3 / Math.max(3, m.verts.length));
}
const partColor = (i) => new THREE.Color().setHSL((i * 0.618034) % 1, 0.62, 0.56);

// ---------------------------------------------------------------- worker
class DecompWorker {
  constructor() {
    this.worker = new Worker("./worker.js", { type: "module" });
    this.pending = null; // { id, resolve, reject, onProgress }
    this.nextId = 1;
    this.ready = new Promise((resolve, reject) => {
      this.worker.onmessage = (e) => {
        const m = e.data;
        if (m.type === "ready") { resolve(m.info); return; }
        if (m.type === "error" && m.id === undefined) { reject(new Error(m.message)); return; }
        const job = this.pending;
        if (!job || m.id !== job.id) return;
        if (m.type === "progress") job.onProgress?.(m);
        else if (m.type === "result") { this.pending = null; job.resolve(m); }
        else if (m.type === "error") { this.pending = null; const err = new Error(m.message); err.aborted = !!m.aborted; job.reject(err); }
      };
      this.worker.onerror = (e) => reject(new Error(e.message ?? "worker failed to start"));
    });
    this.worker.postMessage({ type: "init" });
  }
  decompose(mesh, opts, onProgress) {
    if (this.pending) this.cancel();
    const id = this.nextId++;
    // copies: the mesh stays usable on this thread
    const verts = mesh.verts.slice(), tris = mesh.tris.slice();
    return new Promise((resolve, reject) => {
      this.pending = { id, resolve, reject, onProgress };
      this.worker.postMessage({ type: "decompose", id, verts, tris, opts }, [verts.buffer, tris.buffer]);
    });
  }
  cancel() { this.worker.postMessage({ type: "cancel" }); }
}

// ---------------------------------------------------------------- app
const EXAMPLES = {
  "stairs (procedural)": { kind: "proc" },
  "49160.stl (repo fixture)": { url: "./models/49160.stl" },
  "teapot (voxel remesh)": { url: "./models/teapot_vox64.stl.gz" },
  "bunny (voxel remesh)": { url: "./models/bunny_vox64.stl.gz" },
};

class App {
  constructor() {
    this.world = new World();
    this.worker = new DecompWorker();
    this.wasm = null;
    this.mesh = null; // current plain mesh
    this.base = null; // as loaded (for reset)
    this.undo = [];
    this.result = null;
    this.running = false;
    this.params = {
      example: "bunny (voxel remesh)",
      threshold: 0.1, width: 30, concaveEdges: 16, maxIters: 100, mergeHulls: true, refineRounds: 2,
      autoRun: true,
      tool: "subtract", shape: "sphere", size: 0.35, twist: 0,
      show: "hulls", explode: 0, opacity: 1, showInput: false,
    };
    this.buildGui();
    this.bindPointer();
    this.bindDrop();
    this.boot();
  }

  async boot() {
    status.set("starting WebGPU worker and manifold…", 0);
    const [info, wasm] = await Promise.all([
      this.worker.ready.catch((e) => { status.set(e.message); return null; }),
      Module().then((w) => { w.setup(); return w; }).catch((e) => { status.set(`manifold-3d failed to load: ${e.message}`); return null; }),
    ]);
    this.wasm = wasm;
    this.gpuInfo = info;
    if (info) status.set(`WebGPU ready: ${info.description || info.vendor}`, 0);
    await this.loadExample(this.params.example);
  }

  // ---- GUI
  buildGui() {
    const p = this.params;
    const gui = new GUI({ title: "cuacd-web" });
    const fm = gui.addFolder("Model");
    fm.add(p, "example", Object.keys(EXAMPLES)).name("example").onChange((v) => this.loadExample(v));
    fm.add({ open: () => this.pickFile() }, "open").name("open file…");
    fm.add({ reset: () => this.setMesh(this.base, "reset") }, "reset").name("reset to loaded mesh");
    fm.add({ undo: () => this.undoEdit() }, "undo").name("undo edit");

    const fe = gui.addFolder("Edit (manifold-3d)");
    fe.add(p, "tool", { "none (orbit)": "none", "add primitive": "add", "subtract primitive": "subtract", "intersect primitive": "intersect" }).name("click tool").onChange(() => this.updateToolPreview());
    fe.add(p, "shape", ["sphere", "box", "cylinder"]).name("shape").onChange(() => this.updateToolPreview());
    fe.add(p, "size", 0.05, 1.5, 0.01).name("size (× model)").onChange(() => this.updateToolPreview());
    fe.add(p, "twist", -180, 180, 1).name("twist (deg)").onFinishChange((deg) => this.applyTwist(deg));
    fe.add({ hull: () => this.applyEdit("hull", () => hullOf(this.wasm, this.mesh)) }, "hull").name("replace with convex hull");

    const fd = gui.addFolder("Decomposition");
    fd.add(p, "threshold", 0.01, 0.3, 0.005).name("threshold");
    fd.add(p, "width", 3, 60, 3).name("width (cuts/level)");
    fd.add(p, "concaveEdges", 0, 32, 1).name("concave edges");
    fd.add(p, "refineRounds", 0, 4, 1).name("refine rounds");
    fd.add(p, "maxIters", 1, 200, 1).name("max iterations");
    fd.add(p, "mergeHulls").name("merge hulls");
    fd.add(p, "autoRun").name("re-run after edits");
    fd.add({ run: () => this.run() }, "run").name("▶ decompose");
    fd.add({ cancel: () => this.worker.cancel() }, "cancel").name("■ cancel");

    const fv = gui.addFolder("View");
    fv.add(p, "show", { "convex hulls": "hulls", "part meshes": "parts", "hulls + parts": "both" }).name("show").onChange(() => this.refreshView());
    fv.add(p, "explode", 0, 2, 0.01).name("explode").onChange(() => this.refreshView());
    fv.add(p, "opacity", 0.1, 1, 0.05).name("opacity").onChange(() => this.refreshView());
    fv.add(p, "showInput").name("input wireframe").onChange(() => this.refreshView());
    this.gui = gui;
  }

  // ---- loading
  async loadExample(name) {
    const ex = EXAMPLES[name];
    try {
      let mesh;
      if (ex.kind === "proc") mesh = stairs();
      else {
        status.set(`loading ${name}…`, 0);
        const resp = await fetch(ex.url);
        if (!resp.ok) throw new Error(`${ex.url}: ${resp.status}`);
        let buf;
        if (ex.url.endsWith(".gz")) buf = await new Response(resp.body.pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
        else buf = await resp.arrayBuffer();
        mesh = parseSTL(buf);
      }
      this.base = mesh;
      this.undo = [];
      this.params.twist = 0;
      this.setMesh(mesh, name);
    } catch (e) { status.set(`could not load ${name}: ${e.message}`); }
  }
  pickFile() {
    const input = document.createElement("input");
    input.type = "file"; input.accept = ".obj,.stl,.ply,.glb,.gltf";
    input.onchange = () => input.files[0] && this.loadFile(input.files[0]);
    input.click();
  }
  async loadFile(file) {
    const ext = file.name.toLowerCase().split(".").pop();
    try {
      let mesh;
      if (ext === "obj") mesh = parseOBJ(await file.text());
      else if (ext === "stl") mesh = parseSTL(await file.arrayBuffer());
      else if (ext === "ply") mesh = fromThreeGeometry(new PLYLoader().parse(await file.arrayBuffer()));
      else if (ext === "glb" || ext === "gltf") {
        const gltf = await new GLTFLoader().parseAsync(await file.arrayBuffer(), "");
        let g = null;
        gltf.scene.updateMatrixWorld(true);
        gltf.scene.traverse((o) => { if (o.isMesh && !g) { g = o.geometry.clone().applyMatrix4(o.matrixWorld); } });
        if (!g) throw new Error("no mesh in file");
        mesh = fromThreeGeometry(g);
      } else throw new Error(`unsupported file type .${ext}`);
      this.base = mesh; this.undo = []; this.params.twist = 0;
      this.setMesh(mesh, file.name);
    } catch (e) { status.set(`could not load ${file.name}: ${e.message}`); }
  }
  bindDrop() {
    const drop = document.getElementById("drop");
    addEventListener("dragover", (e) => { e.preventDefault(); drop.style.display = "flex"; });
    addEventListener("dragleave", (e) => { if (!e.relatedTarget) drop.style.display = "none"; });
    addEventListener("drop", (e) => { e.preventDefault(); drop.style.display = "none"; const f = e.dataTransfer.files[0]; if (f) this.loadFile(f); });
  }

  // ---- mesh state
  setMesh(mesh, name) {
    if (!mesh) return;
    this.mesh = mesh;
    this.result = null;
    this.world.partsGroup.clear();
    this.world.inputGroup.clear();
    const g = toGeometry(mesh);
    this.inputMesh = new THREE.Mesh(g, new THREE.MeshStandardMaterial({ color: 0x8a97ad, flatShading: true, roughness: 0.8 }));
    this.inputWire = new THREE.LineSegments(new THREE.WireframeGeometry(g), new THREE.LineBasicMaterial({ color: 0x4db6ff, transparent: true, opacity: 0.35 }));
    this.inputWire.visible = false;
    this.world.inputGroup.add(this.inputMesh, this.inputWire);
    this.modelRadius = g.boundingSphere.radius;
    this.modelCenter = g.boundingSphere.center.clone();
    if (name !== "edit" && name !== "reset") this.world.frame(this.modelRadius, this.modelCenter);
    const closed = isClosed(mesh);
    const info = this.wasm ? inspect(this.wasm, mesh) : null;
    status.set(`${name}: ${nv(mesh)} vertices, ${nt(mesh)} triangles${closed ? "" : " — NOT closed"}`, 0,
      info && Number.isFinite(info.volume) ? `volume ${info.volume.toPrecision(4)}, genus ${info.genus}` : (info ? info.status : ""));
    this.gui.controllers.forEach((c) => c.updateDisplay());
    this.refreshView();
    if (closed && this.params.autoRun) this.run();
    this.updateToolPreview();
  }
  applyEdit(label, fn) {
    if (!this.wasm) { status.set("manifold-3d is not loaded"); return; }
    if (!this.mesh) return;
    try {
      const t0 = performance.now();
      const out = fn();
      this.undo.push(this.mesh);
      if (this.undo.length > 20) this.undo.shift();
      this.setMesh(out, "edit");
      status.sub.textContent += ` · ${label} in ${(performance.now() - t0).toFixed(0)} ms`;
    } catch (e) { status.set(`${label} failed: ${e.message}`); }
  }
  undoEdit() {
    const m = this.undo.pop();
    if (m) this.setMesh(m, "edit");
  }
  applyTwist(deg) {
    // twist is applied to the mesh before the twist (the last non-twist state)
    if (!this.twistSource) this.twistSource = this.mesh;
    const src = this.twistSource;
    this.applyEdit(`twist ${deg}°`, () => (deg === 0 ? src : twist(this.wasm, src, deg, 1)));
    this.twistSource = src;
  }

  // ---- click tools
  bindPointer() {
    const el = this.world.domElement;
    const ray = new THREE.Raycaster();
    const ndc = new THREE.Vector2();
    let down = null;
    const hit = (e) => {
      if (!this.inputMesh) return null;
      ndc.set((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
      ray.setFromCamera(ndc, this.world.camera);
      const targets = this.world.inputGroup.visible ? [this.inputMesh] : this.world.partsGroup.children;
      const h = ray.intersectObjects(targets, false)[0];
      return h ? h.point : null;
    };
    el.addEventListener("pointerdown", (e) => { down = [e.clientX, e.clientY]; });
    el.addEventListener("pointermove", (e) => {
      if (this.params.tool !== "none" && this.preview) {
        const p = hit(e);
        this.preview.visible = !!p;
        if (p) this.preview.position.copy(p);
      }
      this.hover(e);
    });
    el.addEventListener("pointerup", (e) => {
      if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) { down = null; return; }
      down = null;
      if (this.params.tool === "none") return;
      const p = hit(e);
      if (!p) return;
      const size = this.params.size * this.modelRadius;
      const op = this.params.tool;
      this.twistSource = null;
      this.applyEdit(`${op} ${this.params.shape}`, () => booleanWithPrimitive(this.wasm, this.mesh, op, this.params.shape, [p.x, p.y, p.z], size));
    });
  }
  updateToolPreview() {
    this.world.toolGroup.clear();
    this.preview = null;
    if (this.params.tool === "none" || !this.modelRadius) return;
    const s = this.params.size * this.modelRadius;
    const geo = this.params.shape === "sphere" ? new THREE.SphereGeometry(s / 2, 24, 16)
      : this.params.shape === "cylinder" ? new THREE.CylinderGeometry(s / 2, s / 2, s, 24).rotateX(Math.PI / 2)
      : new THREE.BoxGeometry(s, s, s);
    const color = this.params.tool === "add" ? 0x7cf2c4 : this.params.tool === "subtract" ? 0xff7c7c : 0xffd27c;
    this.preview = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.35, depthWrite: false }));
    this.preview.visible = false;
    this.world.toolGroup.add(this.preview);
  }
  hover(e) {
    const tip = document.getElementById("tip");
    if (!this.result || this.world.inputGroup.visible) { tip.style.display = "none"; return; }
    const ndc = new THREE.Vector2((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
    const ray = new THREE.Raycaster();
    ray.setFromCamera(ndc, this.world.camera);
    const h = ray.intersectObjects(this.world.partsGroup.children, false)[0];
    if (!h || h.object.userData.part === undefined) { tip.style.display = "none"; return; }
    const p = this.result.parts[h.object.userData.part];
    tip.style.display = "block";
    tip.style.left = `${e.clientX + 14}px`; tip.style.top = `${e.clientY + 14}px`;
    const cost = Math.max(0.3 * Math.cbrt((3 / (4 * Math.PI)) * Math.max(p.hullVol - p.meshVol, 0)), p.hausdorff ?? 0);
    tip.textContent = `part ${h.object.userData.part}: cost ${cost.toFixed(4)}, hull vol ${p.hullVol.toPrecision(3)}, mesh vol ${p.meshVol.toPrecision(3)}${p.merged > 1 ? `, merged from ${p.merged}` : ""}`;
  }

  // ---- decomposition
  async run() {
    if (!this.mesh || !this.gpuInfo) return;
    const p = this.params;
    const opts = {
      threshold: p.threshold, width: p.width, nConcaveEdges: p.concaveEdges, maxIters: p.maxIters,
      mergeHulls: p.mergeHulls, refineRounds: p.refineRounds,
    };
    this.running = true;
    const t0 = performance.now();
    try {
      const r = await this.worker.decompose(this.mesh, opts, (m) => {
        status.set(m.message, m.fraction, `${m.phase} · ${((performance.now() - t0) / 1000).toFixed(1)} s · ${m.nParts} parts${m.nCutting ? `, cutting ${m.nCutting}` : ""}${m.worstCost ? `, worst cost ${m.worstCost.toFixed(4)}` : ""}`);
      });
      this.result = r;
      const merged = r.merge ? `, ${r.merge.merges} merged in ${r.merge.rounds} rounds` : "";
      status.set(`${r.parts.length} convex parts in ${r.iterations} iterations${merged}`, 1,
        `${(r.stats.wallMs / 1000).toFixed(2)} s · GPU ${(r.stats.gpuMs / 1000).toFixed(2)} s · hover a part for its cost`);
      this.buildParts();
      this.refreshView();
    } catch (e) {
      if (e.aborted) status.set("cancelled", 0);
      else status.set(`decomposition failed: ${e.message}`, 0);
    } finally { this.running = false; }
  }
  buildParts() {
    const g = this.world.partsGroup;
    g.clear();
    this.result.parts.forEach((p, i) => {
      const color = partColor(i);
      const hull = new THREE.Mesh(toGeometry(p.hull), new THREE.MeshStandardMaterial({ color, flatShading: true, roughness: 0.55, metalness: 0.05, transparent: true }));
      const part = new THREE.Mesh(toGeometry(p.mesh), new THREE.MeshStandardMaterial({ color: color.clone().multiplyScalar(0.8), flatShading: true, roughness: 0.9, transparent: true }));
      hull.userData = { part: i, kind: "hull", offset: centroid(p.hull).sub(this.modelCenter) };
      part.userData = { part: i, kind: "part", offset: hull.userData.offset };
      g.add(hull, part);
    });
  }
  refreshView() {
    const p = this.params;
    const has = !!this.result;
    this.world.inputGroup.visible = !has;
    this.inputWire && (this.inputWire.visible = p.showInput && has);
    this.inputMesh && (this.inputMesh.visible = !has);
    if (has && p.showInput) { this.world.inputGroup.visible = true; this.inputMesh.visible = false; }
    for (const o of this.world.partsGroup.children) {
      o.visible = p.show === "both" || (p.show === "hulls") === (o.userData.kind === "hull");
      o.material.opacity = o.userData.kind === "hull" && p.show === "both" ? Math.min(p.opacity, 0.45) : p.opacity;
      o.material.depthWrite = o.material.opacity >= 0.99;
      o.position.copy(o.userData.offset).multiplyScalar(p.explode);
    }
  }
}

/** three.js geometry → plain welded mesh */
function fromThreeGeometry(geo) {
  const g = geo.index ? geo.toNonIndexed() : geo;
  return weld(g.getAttribute("position").array);
}

new App();
