import { HAUSDORFF_SHADER } from "./hausdorff_shader.js";
// shaders.ts — WGSL kernels for the lookahead tree search.
//
// Data model (all offsets are element indices into one u32 "pool" buffer; the
// float regions are bitcast on access so a single storage binding covers
// vertices, triangles and boundary edges):
//
//   pool[vBase ..)  vertex floats, 3 per vertex
//   pool[tBase ..)  triangle indices, 3 per triangle, local to the part
//   pool[eBase ..)  boundary edges, 8 words each: p.xyz q.xyz planeId pad
//
// Parts produced by a cut are *uncapped*: the plane leaves an open loop on
// each half. Volume stays exact because the cap of a planar loop contributes
// zero flux when the divergence-theorem anchor lies in that plane, and a
// closed-form correction term handles the planes the anchor is not on (see
// `edge_term`). The convex hull of an uncapped half equals the hull of the
// capped half because capping adds no vertices, so the rv cost is unchanged.
//
// Kernels, run once per tree level:
//   plan    one workgroup per (item, cut): bbox, plane, output counts
//   scan    one workgroup: prefix sums → pool offsets, child parts and items
//   emit    one workgroup per job: write both halves + volumes
//   hull    one workgroup per child part: hull volume (incremental hull)
//   record  one thread per child item: sort parts by rv, record level cost
export const WG = 256;
export const VIS_MAX = 2048;
const COMMON = /* wgsl */ `
struct Params {
  itemsInBase: u32, nItemsIn: u32, itemsOutBase: u32, jobBase: u32,
  width: u32, nEdgeCuts: u32, edgeStride: u32, totalWidth: u32,
  minEdgeDist: f32, level: u32, totalLevels: u32, partBase: u32,
  vBase: u32, vCap: u32, tBase: u32, tCap: u32,
  eBase: u32, eCap: u32, sCap: u32, edgePlaneBase: u32,
  hullChunkBase: u32, hullSlabWords: u32, unused22: f32, nChildParts: u32,
  nJobs: u32, hullSlabBase: u32, hullChunkSize: u32, exportBase: u32,
  exportStride: u32, lastLevel: u32, pad4: u32, nSeeds: u32,
}

struct PartDesc {
  vOff: u32, nv: u32, tOff: u32, nt: u32,
  eOff: u32, ne: u32, meshVol: f32, hullVol: f32,
  planeIds: vec4<u32>,
  nPlanes: u32, flags: u32, pad0: u32, pad1: u32,
}

struct ItemDesc {
  partIds: array<u32, 8>,
  np: u32, srcPart: u32, initialCut: u32, nLevels: u32,
  levelCosts: array<f32, 4>,
  flags: u32, parentItem: u32, cutIdx: u32, pad0: u32,
}

struct Job {
  plane: vec4<f32>,
  parentItem: u32, parentPart: u32, valid: u32, cutIdx: u32,
  cnt: array<u32, 8>,   // posNv posNt posNe negNv negNt negNe posKept negKept
  off: array<u32, 8>,   // posV posT posE negV negT negE scratch pad
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> pool: array<u32>;
@group(0) @binding(2) var<storage, read_write> parts: array<PartDesc>;
@group(0) @binding(3) var<storage, read_write> items: array<ItemDesc>;
@group(0) @binding(4) var<storage, read_write> jobs: array<Job>;
@group(0) @binding(5) var<storage, read_write> planes: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> scratch: array<u32>;
@group(0) @binding(7) var<storage, read_write> ctr: array<atomic<u32>>;

const CTR_V = 0u; const CTR_T = 1u; const CTR_E = 2u; const CTR_S = 3u;
const CTR_OVERFLOW = 4u; const CTR_HULLFAIL = 5u; const CTR_NPOVER = 6u; const CTR_PRUNED = 7u;
const NONE = 0xffffffffu;
const K_RV = 0.3;
const PI = 3.14159265358979;

fn wg_linear(wg: vec3<u32>) -> u32 { return wg.y * 65535u + wg.x; }

fn vert(off: u32, i: u32) -> vec3<f32> {
  let b = P.vBase + off + 3u * i;
  return vec3<f32>(bitcast<f32>(pool[b]), bitcast<f32>(pool[b + 1u]), bitcast<f32>(pool[b + 2u]));
}
fn put_vert(off: u32, i: u32, p: vec3<f32>) {
  let b = P.vBase + off + 3u * i;
  pool[b] = bitcast<u32>(p.x); pool[b + 1u] = bitcast<u32>(p.y); pool[b + 2u] = bitcast<u32>(p.z);
}
fn tri(off: u32, t: u32) -> vec3<u32> {
  let b = P.tBase + off + 3u * t;
  return vec3<u32>(pool[b], pool[b + 1u], pool[b + 2u]);
}
fn put_tri(off: u32, t: u32, a: u32, b: u32, c: u32) {
  if (P.lastLevel != 0u) { return; }
  let k = P.tBase + off + 3u * t;
  pool[k] = a; pool[k + 1u] = b; pool[k + 2u] = c;
}
fn edge_p(off: u32, e: u32) -> vec3<f32> {
  let b = P.eBase + 8u * (off + e);
  return vec3<f32>(bitcast<f32>(pool[b]), bitcast<f32>(pool[b + 1u]), bitcast<f32>(pool[b + 2u]));
}
fn edge_q(off: u32, e: u32) -> vec3<f32> {
  let b = P.eBase + 8u * (off + e) + 3u;
  return vec3<f32>(bitcast<f32>(pool[b]), bitcast<f32>(pool[b + 1u]), bitcast<f32>(pool[b + 2u]));
}
fn edge_plane(off: u32, e: u32) -> u32 { return pool[P.eBase + 8u * (off + e) + 6u]; }
fn put_edge(off: u32, e: u32, p: vec3<f32>, q: vec3<f32>, pl: u32) {
  if (P.lastLevel != 0u) { return; }
  let b = P.eBase + 8u * (off + e);
  pool[b] = bitcast<u32>(p.x); pool[b + 1u] = bitcast<u32>(p.y); pool[b + 2u] = bitcast<u32>(p.z);
  pool[b + 3u] = bitcast<u32>(q.x); pool[b + 4u] = bitcast<u32>(q.y); pool[b + 5u] = bitcast<u32>(q.z);
  pool[b + 6u] = pl; pool[b + 7u] = 0u;
}

fn plane_point(pl: vec4<f32>) -> vec3<f32> {
  let n2 = max(dot(pl.xyz, pl.xyz), 1e-30);
  return -pl.w * pl.xyz / n2;
}

// Divergence-theorem anchor for a part bounded by the given planes. If the
// anchor lies on every plane, every open loop contributes zero and the volume
// of the uncapped mesh is exact. Two planes: a point on their line. Three
// planes: their common point. Parallel pairs fall back to the newest plane
// and rely on the per-edge correction for the others.
fn anchor_for(plIn: array<vec4<f32>, 4>, n: u32) -> vec3<f32> {
  var pl = plIn;
  if (n == 0u) { return vec3<f32>(0.0); }
  if (n == 1u) { return plane_point(pl[0]); }
  if (n >= 3u) {
    let n1 = pl[n - 3u].xyz; let n2 = pl[n - 2u].xyz; let n3 = pl[n - 1u].xyz;
    let det = dot(n1, cross(n2, n3));
    if (abs(det) > 1e-4) {
      let e1 = -pl[n - 3u].w; let e2 = -pl[n - 2u].w; let e3 = -pl[n - 1u].w;
      return (e1 * cross(n2, n3) + e2 * cross(n3, n1) + e3 * cross(n1, n2)) / det;
    }
  }
  let a = pl[n - 2u]; let b = pl[n - 1u];
  let L = cross(a.xyz, b.xyz);
  let l2 = dot(L, L);
  if (l2 > 1e-8) {
    let e1 = -a.w; let e2 = -b.w;
    return (e1 * cross(b.xyz, L) + e2 * cross(L, a.xyz)) / l2;
  }
  return plane_point(b);
}

fn tri_flux(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, o: vec3<f32>) -> f32 {
  return dot(a - o, cross(b - o, c - o)) / 6.0;
}
// Flux of the implicit cap triangle (c, q, p) closing open edge p->q, where c
// is any point in the edge's plane. Zero when the anchor o lies in that plane.
fn edge_term(p: vec3<f32>, q: vec3<f32>, c: vec3<f32>, o: vec3<f32>) -> f32 {
  return dot(c - o, cross(q - o, p - o)) / 6.0;
}

fn rv_cost(meshVol: f32, hullVol: f32) -> f32 {
  let gap = max(hullVol - meshVol, 0.0);
  return K_RV * pow((3.0 / (4.0 * PI)) * gap, 1.0 / 3.0);
}



// Three-state vertex sign relative to the plane (PC_EPS as in plane_cut.cuh).
const PC_EPS = 1e-6;
fn vsign(d: f32) -> i32 { return select(select(0, -1, d < -PC_EPS), 1, d > PC_EPS); }

// Sutherland–Hodgman clip of one triangle against a half-space. side is +1
// for the positive half, -1 for the negative half. Entries of src are
// 0..2 (original vertex k) or 4+k (intersection on edge k→k+1). On-plane
// vertices are kept by both sides; intersections only occur on edges whose
// endpoints have strictly opposite signs, so no degenerate pieces arise.
struct Poly { n: u32, src: array<u32, 4> }
fn clip_tri(sg: vec3<i32>, side: i32) -> Poly {
  var p: Poly;
  p.n = 0u;
  for (var k = 0u; k < 3u; k++) {
    let q = (k + 1u) % 3u;
    let sp = sg[k] * side; let sq = sg[q] * side;
    if (sp >= 0) { p.src[p.n] = k; p.n += 1u; }
    if (sg[k] != 0 && sg[q] != 0 && sp * sq < 0) { p.src[p.n] = 4u + k; p.n += 1u; }
  }
  return p;
}
struct PolyStats { ntri: u32, nnew: u32, nedge: u32 }
// Triangles (fan), intersection vertices and on-plane edges produced by a polygon.
fn poly_stats(p: Poly, sg: vec3<i32>) -> PolyStats {
  var st: PolyStats;
  st.ntri = select(0u, p.n - 2u, p.n >= 3u);
  st.nnew = 0u; st.nedge = 0u;
  if (p.n >= 3u) {
    for (var i = 0u; i < p.n; i++) {
      let u = p.src[i]; let v = p.src[(i + 1u) % p.n];
      if (u >= 4u) { st.nnew += 1u; }
      let uOn = u >= 4u || sg[u] == 0;
      let vOn = v >= 4u || sg[v] == 0;
      if (uOn && vOn) { st.nedge += 1u; }
    }
  }
  return st;
}

// ---- workgroup helpers (256 threads) ----
var<workgroup> sh_u: array<u32, 256>;
var<workgroup> sh_f: array<f32, 256>;
var<workgroup> sh_u1: u32;
var<workgroup> sh_f1: f32;
var<workgroup> sh_v1: vec3<f32>;
var<workgroup> sh_v2: vec3<f32>;
var<workgroup> sh_pl: vec4<f32>;
var<workgroup> sh_dims: vec4<u32>;

fn reduce_u(v: u32, lid: u32) -> u32 {
  sh_u[lid] = v;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s = s >> 1u) {
    if (lid < s) { sh_u[lid] = sh_u[lid] + sh_u[lid + s]; }
    workgroupBarrier();
  }
  let r = sh_u[0];
  workgroupBarrier();
  return r;
}
fn reduce_f(v: f32, lid: u32) -> f32 {
  sh_f[lid] = v;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s = s >> 1u) {
    if (lid < s) { sh_f[lid] = sh_f[lid] + sh_f[lid + s]; }
    workgroupBarrier();
  }
  let r = sh_f[0];
  workgroupBarrier();
  return r;
}
fn reduce_min_f(v: f32, lid: u32) -> f32 {
  sh_f[lid] = v;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s = s >> 1u) {
    if (lid < s) { sh_f[lid] = min(sh_f[lid], sh_f[lid + s]); }
    workgroupBarrier();
  }
  let r = sh_f[0];
  workgroupBarrier();
  return r;
}
fn reduce_max_f(v: f32, lid: u32) -> f32 {
  sh_f[lid] = v;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s = s >> 1u) {
    if (lid < s) { sh_f[lid] = max(sh_f[lid], sh_f[lid + s]); }
    workgroupBarrier();
  }
  let r = sh_f[0];
  workgroupBarrier();
  return r;
}
// argmax with lowest-index tie break; returns index, value in sh_f1
fn argmax_f(v: f32, idx: u32, lid: u32) -> u32 {
  sh_f[lid] = v; sh_u[lid] = idx;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s = s >> 1u) {
    if (lid < s) {
      let a = sh_f[lid]; let b = sh_f[lid + s];
      let ia = sh_u[lid]; let ib = sh_u[lid + s];
      if (b > a || (b == a && ib < ia)) { sh_f[lid] = b; sh_u[lid] = ib; }
    }
    workgroupBarrier();
  }
  let r = sh_u[0];
  if (lid == 0u) { sh_f1 = sh_f[0]; }
  workgroupBarrier();
  return r;
}
struct ScanR { prefix: u32, total: u32 }
// exclusive prefix sum over the workgroup
fn scan_u(v: u32, lid: u32) -> ScanR {
  sh_u[lid] = v;
  workgroupBarrier();
  for (var off = 1u; off < 256u; off = off << 1u) {
    var t = 0u;
    if (lid >= off) { t = sh_u[lid - off]; }
    workgroupBarrier();
    sh_u[lid] = sh_u[lid] + t;
    workgroupBarrier();
  }
  let incl = sh_u[lid];
  let total = sh_u[255];
  workgroupBarrier();
  return ScanR(incl - v, total);
}
fn bbox_reduce(lo: vec3<f32>, hi: vec3<f32>, lid: u32) {
  let lx = reduce_min_f(lo.x, lid); let ly = reduce_min_f(lo.y, lid); let lz = reduce_min_f(lo.z, lid);
  let hx = reduce_max_f(hi.x, lid); let hy = reduce_max_f(hi.y, lid); let hz = reduce_max_f(hi.z, lid);
  if (lid == 0u) { sh_v1 = vec3<f32>(lx, ly, lz); sh_v2 = vec3<f32>(hx, hy, hz); }
  workgroupBarrier();
}
`;
const PLAN = /* wgsl */ `
@compute @workgroup_size(256)
fn plan(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
  let lid = lid3.x;
  let g = wg_linear(wg);
  if (g >= P.nJobs) { return; }
  let itemIdx = g / P.totalWidth;
  let cut = g % P.totalWidth;
  let jobIdx = P.jobBase + g;

  if (lid == 0u) {
    let it = items[P.itemsInBase + itemIdx];
    sh_u1 = select(0u, 1u, it.flags == 1u && it.np > 0u);
    jobs[jobIdx].valid = 0u;
    jobs[jobIdx].parentItem = itemIdx;
    jobs[jobIdx].cutIdx = cut;
    for (var k = 0u; k < 8u; k++) { jobs[jobIdx].cnt[k] = 0u; jobs[jobIdx].off[k] = 0u; }
    planes[jobIdx] = vec4<f32>(0.0);
  }
  let itemOk = workgroupUniformLoad(&sh_u1);
  if (itemOk == 0u) { return; }

  var it = items[P.itemsInBase + itemIdx];
  let partId = it.partIds[it.np - 1u];
  let part = parts[partId];
  if (lid == 0u) { jobs[jobIdx].parentPart = partId; sh_dims = vec4<u32>(part.nv, part.nt, part.ne, 0u); }
  let dims = workgroupUniformLoad(&sh_dims);
  let unv = dims.x; let unt = dims.y; let une = dims.z;

  // bounding box
  var lo = vec3<f32>(1e30); var hi = vec3<f32>(-1e30);
  for (var i = lid; i < unv; i += 256u) {
    let p = vert(part.vOff, i);
    lo = min(lo, p); hi = max(hi, p);
  }
  bbox_reduce(lo, hi, lid);
  let blo = workgroupUniformLoad(&sh_v1);
  let bhi = workgroupUniformLoad(&sh_v2);
  let ext = bhi - blo;
  let eps = 1e-6;
  let lim = 2.0 * P.minEdgeDist + eps;
  if (ext.x <= lim && ext.y <= lim && ext.z <= lim) {
    // too small to cut on every axis: the item becomes a leaf with zero
    // cost for the remaining levels (mirrors la_expand's extra_leaves)
    if (cut == 0u && lid == 0u) {
      var it2 = items[P.itemsInBase + itemIdx];
      for (var l = it2.nLevels; l < min(P.totalLevels, 4u); l++) { it2.levelCosts[l] = 0.0; }
      it2.nLevels = min(P.totalLevels, 4u);
      it2.flags = 2u;
      items[P.itemsInBase + itemIdx] = it2;
    }
    return;
  }

  // plane (computed by thread 0, broadcast through workgroup memory so the
  // validity decision is uniform); thread 0 also reserves the scratch
  // region (2 words per parent vertex: side flags, later the remap)
  if (lid == 0u) {
    var pl0 = vec4<f32>(0.0);
    var ok = 1u;
    if (cut >= P.width) {
      pl0 = planes[P.edgePlaneBase + it.srcPart * P.edgeStride + (cut - P.width)];
      ok = select(0u, 1u, dot(pl0.xyz, pl0.xyz) > 0.0);
    } else {
      let cpa = P.width / 3u;
      let axis = cut / cpa;
      let slice = cut % cpa;
      let loV = blo[axis] + P.minEdgeDist;
      let hiV = bhi[axis] - P.minEdgeDist;
      if (loV >= hiV) { ok = 0u; }
      let frac = (f32(slice) + 1.0) / (f32(cpa) + 1.0);
      let pos = loV + frac * (hiV - loV);
      pl0 = vec4<f32>(select(0.0, 1.0, axis == 0u), select(0.0, 1.0, axis == 1u), select(0.0, 1.0, axis == 2u), -pos);
    }
    var sOff = 0u;
    if (ok == 1u) {
      sOff = atomicAdd(&ctr[CTR_S], 2u * unv);
      if (sOff + 2u * unv > P.sCap) { ok = 0u; atomicStore(&ctr[CTR_OVERFLOW], 1u); }
    }
    sh_pl = pl0;
    sh_u1 = ok;
    sh_dims = vec4<u32>(sOff, 0u, 0u, 0u);
  }
  let ok = workgroupUniformLoad(&sh_u1);
  let pl = workgroupUniformLoad(&sh_pl);
  let sOff = workgroupUniformLoad(&sh_dims).x;
  if (ok == 0u) { return; }

  // side flags: scratch[sOff+2v] = referenced by a positive-side polygon,
  // scratch[sOff+2v+1] = by a negative-side polygon (plain 1-stores)
  for (var i = lid; i < unv; i += 256u) { scratch[sOff + 2u * i] = 0u; scratch[sOff + 2u * i + 1u] = 0u; }
  storageBarrier();
  workgroupBarrier();
  var posT = 0u; var negT = 0u; var posNew = 0u; var negNew = 0u; var posE = 0u; var negE = 0u;
  for (var t = lid; t < unt; t += 256u) {
    let tr = tri(part.tOff, t);
    let d = vec3<f32>(
      dot(pl.xyz, vert(part.vOff, tr.x)) + pl.w,
      dot(pl.xyz, vert(part.vOff, tr.y)) + pl.w,
      dot(pl.xyz, vert(part.vOff, tr.z)) + pl.w);
    let sg = vec3<i32>(vsign(d.x), vsign(d.y), vsign(d.z));
    if (sg.x == 0 && sg.y == 0 && sg.z == 0) { continue; }  // coplanar face: dropped
    let pp = clip_tri(sg, 1);
    let pn = clip_tri(sg, -1);
    let sp = poly_stats(pp, sg);
    let sn = poly_stats(pn, sg);
    posT += sp.ntri; posNew += sp.nnew; posE += sp.nedge;
    negT += sn.ntri; negNew += sn.nnew; negE += sn.nedge;
    if (pp.n >= 3u) { for (var i = 0u; i < pp.n; i++) { let u = pp.src[i]; if (u < 4u) { scratch[sOff + 2u * tr[u]] = 1u; } } }
    if (pn.n >= 3u) { for (var i = 0u; i < pn.n; i++) { let u = pn.src[i]; if (u < 4u) { scratch[sOff + 2u * tr[u] + 1u] = 1u; } } }
  }
  storageBarrier();
  workgroupBarrier();
  var posKept = 0u; var negKept = 0u;
  for (var i = lid; i < unv; i += 256u) {
    posKept += scratch[sOff + 2u * i];
    negKept += scratch[sOff + 2u * i + 1u];
  }
  for (var e = lid; e < une; e += 256u) {
    let dp = dot(pl.xyz, edge_p(part.eOff, e)) + pl.w;
    let dq = dot(pl.xyz, edge_q(part.eOff, e)) + pl.w;
    let sp = dp >= 0.0; let sq = dq >= 0.0;
    if (sp == sq) { if (sp) { posE += 1u; } else { negE += 1u; } }
    else { posE += 1u; negE += 1u; }
  }
  let rPosKept = reduce_u(posKept, lid);
  let rNegKept = reduce_u(negKept, lid);
  let rPosT = reduce_u(posT, lid);
  let rNegT = reduce_u(negT, lid);
  let rPosNew = reduce_u(posNew, lid);
  let rNegNew = reduce_u(negNew, lid);
  let rPosE = reduce_u(posE, lid);
  let rNegE = reduce_u(negE, lid);
  if (lid == 0u) {
    jobs[jobIdx].plane = pl;
    jobs[jobIdx].valid = 1u;
    jobs[jobIdx].cnt[0] = rPosKept + rPosNew;
    jobs[jobIdx].cnt[1] = rPosT;
    jobs[jobIdx].cnt[2] = rPosE;
    jobs[jobIdx].cnt[3] = rNegKept + rNegNew;
    jobs[jobIdx].cnt[4] = rNegT;
    jobs[jobIdx].cnt[5] = rNegE;
    jobs[jobIdx].cnt[6] = rPosKept;
    jobs[jobIdx].cnt[7] = rNegKept;
    jobs[jobIdx].off[6] = sOff;
    planes[jobIdx] = pl;
  }
}
`;
const SCAN = /* wgsl */ `
var<workgroup> sc_run: array<u32, 8>;
var<workgroup> sc_runU: array<u32, 8>;

@compute @workgroup_size(256)
fn scan(@builtin(local_invocation_id) lid3: vec3<u32>) {
  let lid = lid3.x;
  if (lid == 0u) {
    sc_run[0] = atomicLoad(&ctr[CTR_V]);
    sc_run[1] = atomicLoad(&ctr[CTR_T]);
    sc_run[2] = atomicLoad(&ctr[CTR_E]);
    sc_run[3] = atomicLoad(&ctr[CTR_S]);
  }
  workgroupBarrier();
  let nChunks = (P.nJobs + 255u) / 256u;
  for (var c = 0u; c < nChunks; c++) {
    let j = c * 256u + lid;
    let inRange = j < P.nJobs;
    var job: Job;
    var pnv = 0u;
    if (inRange) {
      job = jobs[P.jobBase + j];
    }
    let vNeed = select(0u, 3u * (job.cnt[0] + job.cnt[3]), inRange && job.valid == 1u);
    // leaves of the tree only need vertices (hull) and the volume computed in emit
    let keepTopo = P.lastLevel == 0u;
    let tNeed = select(0u, 3u * (job.cnt[1] + job.cnt[4]), inRange && job.valid == 1u && keepTopo);
    let eNeed = select(0u, job.cnt[2] + job.cnt[5], inRange && job.valid == 1u && keepTopo);
    let sNeed = 0u * pnv;
    let base = sc_run;  // snapshot (uniform after barrier)
    let sv = scan_u(vNeed, lid);
    let st = scan_u(tNeed, lid);
    let se = scan_u(eNeed, lid);
    let ss = scan_u(sNeed, lid);
    let vOff = base[0] + sv.prefix; let tOff = base[1] + st.prefix;
    let eOff = base[2] + se.prefix; let sOff = base[3] + ss.prefix;
    var valid = inRange && job.valid == 1u;
    if (valid && (vOff + vNeed > P.vCap || tOff + tNeed > P.tCap || eOff + eNeed > P.eCap || sOff + sNeed > P.sCap)) {
      valid = false;
      atomicStore(&ctr[CTR_OVERFLOW], 1u);
    }
    if (inRange) {
      let posP = P.partBase + 2u * j;
      let negP = posP + 1u;
      let outIdx = P.itemsOutBase + j;
      var it: ItemDesc;
      it.flags = 0u;
      it.np = 0u;
      if (valid) {
        let parent = items[P.itemsInBase + job.parentItem];
        let pp = parts[job.parentPart];
        jobs[P.jobBase + j].off[0] = vOff;
        jobs[P.jobBase + j].off[1] = tOff;
        jobs[P.jobBase + j].off[2] = eOff;
        jobs[P.jobBase + j].off[3] = vOff + 3u * job.cnt[0];
        jobs[P.jobBase + j].off[4] = select(tOff, tOff + 3u * job.cnt[1], keepTopo);
        jobs[P.jobBase + j].off[5] = select(eOff, eOff + job.cnt[2], keepTopo);
        var pd: PartDesc;
        pd.planeIds = pp.planeIds;
        pd.nPlanes = pp.nPlanes;
        if (pd.nPlanes < 4u) { pd.planeIds[pd.nPlanes] = P.jobBase + j; pd.nPlanes += 1u; }
        else { pd.planeIds = vec4<u32>(pd.planeIds.y, pd.planeIds.z, pd.planeIds.w, P.jobBase + j); }
        pd.meshVol = 0.0; pd.hullVol = 0.0; pd.flags = 0u; pd.pad0 = 0u; pd.pad1 = 0u;
        var pos = pd; var neg = pd;
        pos.vOff = vOff; pos.nv = job.cnt[0]; pos.tOff = tOff; pos.nt = select(0u, job.cnt[1], keepTopo); pos.eOff = eOff; pos.ne = select(0u, job.cnt[2], keepTopo);
        neg.vOff = vOff + 3u * job.cnt[0]; neg.nv = job.cnt[3]; neg.tOff = select(tOff, tOff + 3u * job.cnt[1], keepTopo); neg.nt = select(0u, job.cnt[4], keepTopo);
        neg.eOff = select(eOff, eOff + job.cnt[2], keepTopo); neg.ne = select(0u, job.cnt[5], keepTopo);
        // flags bit 1: this part has no stored topology (leaf)
        pos.flags = select(0u, 2u, !keepTopo); neg.flags = pos.flags;
        parts[posP] = pos;
        parts[negP] = neg;
        let bothNonEmpty = job.cnt[0] > 0u && job.cnt[1] > 0u && job.cnt[3] > 0u && job.cnt[4] > 0u;
        it = parent;
        if (parent.np >= 8u) {
          atomicStore(&ctr[CTR_NPOVER], 1u);
          it.flags = 0u;
        } else if (bothNonEmpty) {
          it.partIds[parent.np - 1u] = posP;
          it.partIds[parent.np] = negP;
          it.np = parent.np + 1u;
          it.flags = 1u;
          it.parentItem = job.parentItem;
          it.cutIdx = job.cutIdx;
          if (parent.nLevels == 0u) { it.initialCut = job.cutIdx; }
        } else {
          it.flags = 0u;
          jobs[P.jobBase + j].valid = 2u;  // empty half: cut discarded
        }
      } else {
        jobs[P.jobBase + j].valid = 0u;
        var z: PartDesc;
        parts[posP] = z; parts[negP] = z;
      }
      items[outIdx] = it;
    }
    if (lid == 255u) {
      sc_run[0] = base[0] + sv.total; sc_run[1] = base[1] + st.total;
      sc_run[2] = base[2] + se.total; sc_run[3] = base[3] + ss.total;
    }
    workgroupBarrier();
  }
  if (lid == 0u) {
    atomicStore(&ctr[CTR_V], min(sc_run[0], P.vCap));
    atomicStore(&ctr[CTR_T], min(sc_run[1], P.tCap));
    atomicStore(&ctr[CTR_E], min(sc_run[2], P.eCap));
    atomicStore(&ctr[CTR_S], 0u);  // scratch is per level (reserved by plan)
  }
}
`;
const EMIT = /* wgsl */ `
var<workgroup> em_run: array<u32, 8>;

@compute @workgroup_size(256)
fn emit(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
  let lid = lid3.x;
  let g = wg_linear(wg);
  if (g >= P.nJobs) { return; }
  let jobIdx = P.jobBase + g;
  if (lid == 0u) { sh_u1 = jobs[jobIdx].valid; }
  let valid = workgroupUniformLoad(&sh_u1);
  if (valid != 1u) { return; }
  let job = jobs[jobIdx];
  let part = parts[job.parentPart];
  if (lid == 0u) { sh_dims = vec4<u32>(part.nv, part.nt, part.ne, 0u); }
  let dims = workgroupUniformLoad(&sh_dims);
  let unv = dims.x; let unt = dims.y; let une = dims.z;
  let pl = job.plane;
  let cP = plane_point(pl);

  // anchor from the parent's planes plus the new one
  var pls: array<vec4<f32>, 4>;
  var npl = 0u;
  for (var k = 0u; k < part.nPlanes; k++) { pls[npl] = planes[part.planeIds[k]]; npl += 1u; }
  if (npl < 4u) { pls[npl] = pl; npl += 1u; } else { pls = array<vec4<f32>, 4>(pls[1], pls[2], pls[3], pl); }
  let o = anchor_for(pls, npl);

  let posV = job.off[0]; let posT = job.off[1]; let posE = job.off[2];
  let negV = job.off[3]; let negT = job.off[4]; let negE = job.off[5];
  let sOff = job.off[6];
  let posKept = job.cnt[6]; let negKept = job.cnt[7];

  // 1. vertex remap (flags from plan → compact indices) + copy members
  if (lid == 0u) { em_run[0] = 0u; em_run[1] = 0u; }
  workgroupBarrier();
  let nvChunks = (unv + 255u) / 256u;
  for (var c = 0u; c < nvChunks; c++) {
    let i = c * 256u + lid;
    var fp = 0u; var fn_ = 0u;
    var p = vec3<f32>(0.0);
    if (i < unv) { fp = scratch[sOff + 2u * i]; fn_ = scratch[sOff + 2u * i + 1u]; p = vert(part.vOff, i); }
    let base0 = em_run[0]; let base1 = em_run[1];
    let sp = scan_u(fp, lid);
    let sn = scan_u(fn_, lid);
    if (i < unv) {
      var ip = NONE; var in_ = NONE;
      if (fp == 1u) { ip = base0 + sp.prefix; put_vert(posV, ip, p); }
      if (fn_ == 1u) { in_ = base1 + sn.prefix; put_vert(negV, in_, p); }
      scratch[sOff + 2u * i] = ip; scratch[sOff + 2u * i + 1u] = in_;
    }
    if (lid == 255u) { em_run[0] = base0 + sp.total; em_run[1] = base1 + sn.total; }
    workgroupBarrier();
  }
  storageBarrier();
  workgroupBarrier();

  // 2. triangles
  var fluxPos = 0.0; var fluxNeg = 0.0;
  if (lid == 0u) { for (var k = 0u; k < 6u; k++) { em_run[k] = 0u; } }
  workgroupBarrier();
  let ntChunks = (unt + 255u) / 256u;
  for (var c = 0u; c < ntChunks; c++) {
    let t = c * 256u + lid;
    var tr = vec3<u32>(0u);
    var d = vec3<f32>(0.0);
    var sg = vec3<i32>(0);
    var pp: Poly; var pn: Poly;
    pp.n = 0u; pn.n = 0u;
    var sp: PolyStats; var sn: PolyStats;
    sp.ntri = 0u; sp.nnew = 0u; sp.nedge = 0u; sn = sp;
    var live = false;
    if (t < unt) {
      tr = tri(part.tOff, t);
      d = vec3<f32>(
        dot(pl.xyz, vert(part.vOff, tr.x)) + pl.w,
        dot(pl.xyz, vert(part.vOff, tr.y)) + pl.w,
        dot(pl.xyz, vert(part.vOff, tr.z)) + pl.w);
      sg = vec3<i32>(vsign(d.x), vsign(d.y), vsign(d.z));
      live = !(sg.x == 0 && sg.y == 0 && sg.z == 0);
      if (live) {
        pp = clip_tri(sg, 1); pn = clip_tri(sg, -1);
        sp = poly_stats(pp, sg); sn = poly_stats(pn, sg);
      }
    }
    let bPT = em_run[0]; let bNT = em_run[1]; let bPV = em_run[2]; let bNV = em_run[3]; let bPE = em_run[4]; let bNE = em_run[5];
    let s1 = scan_u(sp.ntri | (sp.nnew << 10u) | (sp.nedge << 20u), lid);
    let s2 = scan_u(sn.ntri | (sn.nnew << 10u) | (sn.nedge << 20u), lid);
    if (live) {
      // positions of the triangle corners
      var pos3: array<vec3<f32>, 3>;
      pos3[0] = vert(part.vOff, tr.x); pos3[1] = vert(part.vOff, tr.y); pos3[2] = vert(part.vOff, tr.z);
      for (var side = 0u; side < 2u; side++) {
        let isPos = side == 0u;
        var poly: Poly;
        if (isPos) { poly = pp; } else { poly = pn; }
        if (poly.n < 3u) { continue; }
        let vOff = select(negV, posV, isPos);
        let tOff = select(negT, posT, isPos);
        let eOff = select(negE, posE, isPos);
        let kept = select(negKept, posKept, isPos);
        let pre = select(s2.prefix, s1.prefix, isPos);
        var tBase = select(bNT, bPT, isPos) + (pre & 1023u);
        var vBase = kept + select(bNV, bPV, isPos) + ((pre >> 10u) & 1023u);
        var eBase = select(bNE, bPE, isPos) + (pre >> 20u);
        // resolve polygon entries to child vertex indices + positions
        var idx: array<u32, 4>;
        var pts: array<vec3<f32>, 4>;
        var onp: array<bool, 4>;
        for (var i = 0u; i < poly.n; i++) {
          let u = poly.src[i];
          if (u < 4u) {
            idx[i] = scratch[sOff + 2u * tr[u] + select(1u, 0u, isPos)];
            pts[i] = pos3[u];
            onp[i] = sg[u] == 0;
          } else {
            let k = u - 4u; let q = (k + 1u) % 3u;
            let tt = d[k] / (d[k] - d[q]);
            let x = pos3[k] + tt * (pos3[q] - pos3[k]);
            put_vert(vOff, vBase, x);
            idx[i] = vBase; vBase += 1u;
            pts[i] = x;
            onp[i] = true;
          }
        }
        var flux = 0.0;
        for (var i = 1u; i + 1u < poly.n; i++) {
          put_tri(tOff, tBase, idx[0], idx[i], idx[i + 1u]);
          tBase += 1u;
          flux += tri_flux(pts[0], pts[i], pts[i + 1u], o);
        }
        for (var i = 0u; i < poly.n; i++) {
          let j = (i + 1u) % poly.n;
          if (onp[i] && onp[j]) {
            put_edge(eOff, eBase, pts[i], pts[j], jobIdx);
            eBase += 1u;
            flux += edge_term(pts[i], pts[j], cP, o);
          }
        }
        if (isPos) { fluxPos += flux; } else { fluxNeg += flux; }
      }
    }
    if (lid == 255u) {
      em_run[0] = bPT + (s1.total & 1023u); em_run[2] = bPV + ((s1.total >> 10u) & 1023u); em_run[4] = bPE + (s1.total >> 20u);
      em_run[1] = bNT + (s2.total & 1023u); em_run[3] = bNV + ((s2.total >> 10u) & 1023u); em_run[5] = bNE + (s2.total >> 20u);
    }
    workgroupBarrier();
  }

  // 3. clip inherited open edges (planes other than the new one)
  // running edge counts continue from em_run[4], em_run[5]
  let neChunks = (une + 255u) / 256u;
  for (var c = 0u; c < neChunks; c++) {
    let e = c * 256u + lid;
    var cP_ = 0u; var cN_ = 0u;
    var p = vec3<f32>(0.0); var q = vec3<f32>(0.0); var pid = 0u;
    var dp = 0.0; var dq = 0.0;
    if (e < une) {
      p = edge_p(part.eOff, e); q = edge_q(part.eOff, e); pid = edge_plane(part.eOff, e);
      dp = dot(pl.xyz, p) + pl.w; dq = dot(pl.xyz, q) + pl.w;
      let sp_ = dp >= 0.0; let sq_ = dq >= 0.0;
      if (sp_ == sq_) { cP_ = select(0u, 1u, sp_); cN_ = 1u - cP_; } else { cP_ = 1u; cN_ = 1u; }
    }
    let bPE = em_run[4]; let bNE = em_run[5];
    let s1 = scan_u(cP_, lid);
    let s2 = scan_u(cN_, lid);
    if (e < une) {
      let cE = plane_point(planes[pid]);
      let sp_ = dp >= 0.0; let sq_ = dq >= 0.0;
      if (sp_ == sq_) {
        if (sp_) { put_edge(posE, bPE + s1.prefix, p, q, pid); fluxPos += edge_term(p, q, cE, o); }
        else { put_edge(negE, bNE + s2.prefix, p, q, pid); fluxNeg += edge_term(p, q, cE, o); }
      } else {
        let tt = dp / (dp - dq);
        let x = p + tt * (q - p);
        if (sp_) {
          put_edge(posE, bPE + s1.prefix, p, x, pid); fluxPos += edge_term(p, x, cE, o);
          put_edge(negE, bNE + s2.prefix, x, q, pid); fluxNeg += edge_term(x, q, cE, o);
        } else {
          put_edge(negE, bNE + s2.prefix, p, x, pid); fluxNeg += edge_term(p, x, cE, o);
          put_edge(posE, bPE + s1.prefix, x, q, pid); fluxPos += edge_term(x, q, cE, o);
        }
      }
    }
    if (lid == 255u) { em_run[4] = bPE + s1.total; em_run[5] = bNE + s2.total; }
    workgroupBarrier();
  }

  let vp = reduce_f(fluxPos, lid);
  let vn = reduce_f(fluxNeg, lid);
  if (lid == 0u) {
    parts[P.partBase + 2u * g].meshVol = abs(vp);
    parts[P.partBase + 2u * g + 1u].meshVol = abs(vn);
  }
}
`;
const HULL = /* wgsl */ `
const VIS_MAX = ${VIS_MAX}u;
var<workgroup> h_vis: array<u32, VIS_MAX>;   // visible face slots
var<workgroup> h_conn: array<atomic<u32>, VIS_MAX / 32u>;  // bitmask: visible entry is edge-connected to the seed
var<workgroup> h_changed: atomic<u32>;
var<workgroup> h_nvis: atomic<u32>;
var<workgroup> h_nnew: atomic<u32>;
var<workgroup> h_nf: u32;      // face slots used
var<workgroup> h_tmp: u32;
var<workgroup> h_fail: u32;
var<workgroup> h_compacted: u32;
var<workgroup> h_seed: array<u32, 80>;
var<workgroup> h_nalive: u32;
var<workgroup> h_ncand: u32;
var<workgroup> h_meshVol: f32;
var<workgroup> h_bound: vec2<f32>;   // (partial path sum of the parent item, best upper bound of the seed)
var<workgroup> h_red_f: array<f32, 1024>;  // 4 batched argmax rows for the seed search (indices live in h_vis)
const KDOP_AXES = array<vec3<f32>, 40>(
  vec3<f32>(-0.5786893, 0.7696723, 0.2696723),
  vec3<f32>(-0.7696723, 0.2696723, 0.5786893),
  vec3<f32>(-0.2696723, 0.5786893, 0.7696723),
  vec3<f32>(-0.5773503, 0.5773503, 0.5773503),
  vec3<f32>(-0.3090170, 0.9363390, 0.1666667),
  vec3<f32>(0.0000000, 0.7453560, 0.6666667),
  vec3<f32>(0.3090170, 0.9363390, 0.1666667),
  vec3<f32>(0.0000000, 0.9341724, 0.3568221),
  vec3<f32>(-0.3090170, 0.9363390, -0.1666667),
  vec3<f32>(0.3090170, 0.9363390, -0.1666667),
  vec3<f32>(0.0000000, 0.7453560, -0.6666667),
  vec3<f32>(0.0000000, 0.9341724, -0.3568221),
  vec3<f32>(-0.5786893, 0.7696723, -0.2696723),
  vec3<f32>(-0.2696723, 0.5786893, -0.7696723),
  vec3<f32>(-0.7696723, 0.2696723, -0.5786893),
  vec3<f32>(-0.5773503, 0.5773503, -0.5773503),
  vec3<f32>(-0.7453560, 0.6666667, 0.0000000),
  vec3<f32>(-0.9363390, 0.1666667, -0.3090170),
  vec3<f32>(-0.9363390, 0.1666667, 0.3090170),
  vec3<f32>(-0.9341724, 0.3568221, 0.0000000),
  vec3<f32>(0.5786893, 0.7696723, 0.2696723),
  vec3<f32>(0.2696723, 0.5786893, 0.7696723),
  vec3<f32>(0.7696723, 0.2696723, 0.5786893),
  vec3<f32>(0.5773503, 0.5773503, 0.5773503),
  vec3<f32>(-0.1666667, 0.3090170, 0.9363390),
  vec3<f32>(-0.6666667, 0.0000000, 0.7453560),
  vec3<f32>(-0.1666667, -0.3090170, 0.9363390),
  vec3<f32>(-0.3568221, 0.0000000, 0.9341724),
  vec3<f32>(-0.9363390, -0.1666667, 0.3090170),
  vec3<f32>(-0.9363390, -0.1666667, -0.3090170),
  vec3<f32>(-0.7453560, -0.6666667, 0.0000000),
  vec3<f32>(-0.9341724, -0.3568221, 0.0000000),
  vec3<f32>(-0.6666667, 0.0000000, -0.7453560),
  vec3<f32>(-0.1666667, 0.3090170, -0.9363390),
  vec3<f32>(-0.1666667, -0.3090170, -0.9363390),
  vec3<f32>(-0.3568221, 0.0000000, -0.9341724),
  vec3<f32>(0.2696723, 0.5786893, -0.7696723),
  vec3<f32>(0.5786893, 0.7696723, -0.2696723),
  vec3<f32>(0.7696723, 0.2696723, -0.5786893),
  vec3<f32>(0.5773503, 0.5773503, -0.5773503)
);


// ---- exact orientation predicate ------------------------------------------
// Every part is quantized to an integer grid of 2^21 cells across its bbox,
// so all coordinates are integer-valued f32 below 2^22 and all differences
// are exact. The orientation sign of a point against a face is decided in
// f32 when the value clears a rigorous rounding bound and by a 96-bit
// integer determinant otherwise: the incremental hull then never sees an
// inconsistent predicate, which is what broke it on sliver faces before.

struct I96 { l: u32, m: u32, h: u32 }   // two's complement, 3 limbs

fn i96_add(a: I96, b: I96) -> I96 {
  let l = a.l + b.l; let c0 = select(0u, 1u, l < a.l);
  let m1 = a.m + b.m; let c1a = select(0u, 1u, m1 < a.m);
  let m = m1 + c0; let c1b = select(0u, 1u, m < m1);
  return I96(l, m, a.h + b.h + c1a + c1b);
}
fn i96_neg(a: I96) -> I96 {
  let l = ~a.l + 1u; let c0 = select(0u, 1u, l == 0u);
  let m = ~a.m + c0; let c1 = select(0u, 1u, c0 == 1u && m == 0u);
  return I96(l, m, ~a.h + c1);
}
// exact x*y for |x|,|y| < 2^31
fn i96_mul32(x: i32, y: i32) -> I96 {
  let ux = u32(abs(x)); let uy = u32(abs(y));
  let xl = ux & 0xffffu; let xh = ux >> 16u; let yl = uy & 0xffffu; let yh = uy >> 16u;
  let ll = xl * yl; let mid = xl * yh + xh * yl; let hh = xh * yh;
  let lo = ll + (mid << 16u); let c = select(0u, 1u, lo < ll);
  var r = I96(lo, hh + (mid >> 16u) + c, 0u);
  if ((x < 0) != (y < 0)) { r = i96_neg(r); }
  return r;
}
// exact a*y for |a| < 2^63, |y| < 2^31 (schoolbook in 16-bit digits, unrolled)
fn i96_col(p: u32, q: u32, carry: u32) -> vec2<u32> {
  // p + q + carry with the overflow count in .y
  let s1 = p + q; let c1 = select(0u, 1u, s1 < p);
  let s2 = s1 + carry; let c2 = select(0u, 1u, s2 < s1);
  return vec2<u32>(s2, c1 + c2);
}
fn i96_mul(a: I96, y: i32) -> I96 {
  let an = (a.h >> 31u) != 0u;
  var m = a; if (an) { m = i96_neg(a); }
  let uy = u32(abs(y));
  let a0 = m.l & 0xffffu; let a1 = m.l >> 16u; let a2 = m.m & 0xffffu; let a3 = m.m >> 16u;
  let y0 = uy & 0xffffu; let y1 = uy >> 16u;
  var r0 = i96_col(a0 * y0, 0u, 0u);
  var r1 = i96_col(a0 * y1, a1 * y0, (r0.x >> 16u) + (r0.y << 16u));
  var r2 = i96_col(a1 * y1, a2 * y0, (r1.x >> 16u) + (r1.y << 16u));
  var r3 = i96_col(a2 * y1, a3 * y0, (r2.x >> 16u) + (r2.y << 16u));
  var r4 = i96_col(a3 * y1, 0u, (r3.x >> 16u) + (r3.y << 16u));
  let r5 = (r4.x >> 16u) + (r4.y << 16u);
  var r = I96((r0.x & 0xffffu) | ((r1.x & 0xffffu) << 16u), (r2.x & 0xffffu) | ((r3.x & 0xffffu) << 16u), (r4.x & 0xffffu) | (r5 << 16u));
  if (an != (y < 0)) { r = i96_neg(r); }
  return r;
}
// sign of det[e1; e2; d] for integer vectors with components below 2^23
fn orient_exact(e1: vec3<i32>, e2: vec3<i32>, d: vec3<i32>) -> i32 {
  let cx = i96_add(i96_mul32(e1.y, e2.z), i96_neg(i96_mul32(e1.z, e2.y)));
  let cy = i96_add(i96_mul32(e1.z, e2.x), i96_neg(i96_mul32(e1.x, e2.z)));
  let cz = i96_add(i96_mul32(e1.x, e2.y), i96_neg(i96_mul32(e1.y, e2.x)));
  let det = i96_add(i96_add(i96_mul(cx, d.x), i96_mul(cy, d.y)), i96_mul(cz, d.z));
  if ((det.h >> 31u) != 0u) { return -1; }
  if ((det.l | det.m | det.h) == 0u) { return 0; }
  return 1;
}
// filtered orientation: n = e1 x e2 as stored (f32 of exact integer products)
fn orient_filtered(n: vec3<f32>, e1: vec3<f32>, e2: vec3<f32>, d: vec3<f32>) -> i32 {
  let det = dot(n, d);
  let bound = 1e-6 * (abs(e1.x) + abs(e1.y) + abs(e1.z)) * (abs(e2.x) + abs(e2.y) + abs(e2.z)) * (abs(d.x) + abs(d.y) + abs(d.z));
  if (det > bound) { return 1; }
  if (det < -bound) { return -1; }
  return orient_exact(vec3<i32>(e1), vec3<i32>(e2), vec3<i32>(d));
}

struct HullCtx { slab: u32, fbase: u32, fmax: u32, vOff: u32, nv: u32, qlo: vec3<f32>, qs: f32 }

// quantized (integer-valued) coordinates of point i in the part's local grid
fn qvert(ctx: HullCtx, i: u32) -> vec3<f32> {
  return round((vert(ctx.vOff, i) - ctx.qlo) * ctx.qs);
}

// face record: 16 words = ia|alive<<31 ib ic bf | nx ny nz cz | ax ay az bx | by bz cx cy
// (bf = |e1|_1 |e2|_1, the error-bound factor of the f32 orientation test)
const FW = 16u;
const ALIVE = 0x80000000u;
fn face_a(ctx: HullCtx, f: u32) -> vec3<u32> {
  let b = ctx.fbase + FW * f;
  return vec3<u32>(scratch[b] & 0x7fffffffu, scratch[b + 1u], scratch[b + 2u]);
}
fn face_alive(ctx: HullCtx, f: u32) -> bool { return (scratch[ctx.fbase + FW * f] & ALIVE) != 0u; }
// Evaluate p against face f: a positive value (the f32 plane distance, or a
// tiny positive number when only the exact test could decide) means outside
// (visible); negative means inside, on the plane, or a dead face. The sign
// is always exact; the magnitude is for ranking only.
fn face_eval(ctx: HullCtx, f: u32, p: vec3<f32>) -> f32 {
  let k = ctx.fbase + FW * f;
  let w0 = scratch[k];
  if ((w0 & ALIVE) == 0u) { return -1e30; }
  let n = vec3<f32>(bitcast<f32>(scratch[k + 4u]), bitcast<f32>(scratch[k + 5u]), bitcast<f32>(scratch[k + 6u]));
  let a = vec3<f32>(bitcast<f32>(scratch[k + 8u]), bitcast<f32>(scratch[k + 9u]), bitcast<f32>(scratch[k + 10u]));
  let d = p - a;
  let det = dot(n, d);
  let bound = 1e-6 * bitcast<f32>(scratch[k + 3u]) * (abs(d.x) + abs(d.y) + abs(d.z));
  if (det > bound) { return det / max(length(n), 1e-30); }
  if (det < -bound) { return -1.0; }
  let b = vec3<f32>(bitcast<f32>(scratch[k + 11u]), bitcast<f32>(scratch[k + 12u]), bitcast<f32>(scratch[k + 13u]));
  let c = vec3<f32>(bitcast<f32>(scratch[k + 14u]), bitcast<f32>(scratch[k + 15u]), bitcast<f32>(scratch[k + 7u]));
  let sg = orient_exact(vec3<i32>(b - a), vec3<i32>(c - a), vec3<i32>(d));
  return select(-1.0, 1e-30, sg > 0);
}
fn face_write(ctx: HullCtx, f: u32, a: u32, b: u32, c: u32) {
  let pa = qvert(ctx, a); let pb = qvert(ctx, b); let pc = qvert(ctx, c);
  let e1 = pb - pa; let e2 = pc - pa;
  let n = cross(e1, e2);
  let bf = (abs(e1.x) + abs(e1.y) + abs(e1.z)) * (abs(e2.x) + abs(e2.y) + abs(e2.z));
  let k = ctx.fbase + FW * f;
  scratch[k] = a | ALIVE; scratch[k + 1u] = b; scratch[k + 2u] = c; scratch[k + 3u] = bitcast<u32>(bf);
  scratch[k + 4u] = bitcast<u32>(n.x); scratch[k + 5u] = bitcast<u32>(n.y); scratch[k + 6u] = bitcast<u32>(n.z); scratch[k + 7u] = bitcast<u32>(pc.z);
  scratch[k + 8u] = bitcast<u32>(pa.x); scratch[k + 9u] = bitcast<u32>(pa.y); scratch[k + 10u] = bitcast<u32>(pa.z); scratch[k + 11u] = bitcast<u32>(pb.x);
  scratch[k + 12u] = bitcast<u32>(pb.y); scratch[k + 13u] = bitcast<u32>(pb.z); scratch[k + 14u] = bitcast<u32>(pc.x); scratch[k + 15u] = bitcast<u32>(pc.y);
}
fn face_kill(ctx: HullCtx, f: u32) { let k = ctx.fbase + FW * f; scratch[k] = scratch[k] & 0x7fffffffu; }
fn shares_edge(f: vec3<u32>, g: vec3<u32>) -> bool {
  // true when some directed edge of f appears reversed in g (adjacent faces)
  for (var k = 0u; k < 3u; k++) {
    let a = select(select(f.z, f.y, k == 1u), f.x, k == 0u);
    let b = select(select(f.x, f.z, k == 1u), f.y, k == 0u);
    if ((g.x == b && g.y == a) || (g.y == b && g.z == a) || (g.z == b && g.x == a)) { return true; }
  }
  return false;
}
fn conn_get(i: u32) -> bool { return (atomicLoad(&h_conn[i >> 5u]) & (1u << (i & 31u))) != 0u; }

// Remove dead face slots (uniform control flow; all threads participate).
fn hull_compact(ctx: HullCtx, lid: u32) {
  let nf = workgroupUniformLoad(&h_nf);
  var run = 0u;
  let nChunks = (nf + 255u) / 256u;
  for (var c = 0u; c < nChunks; c++) {
    let f = c * 256u + lid;
    var alive = 0u;
    var w = array<u32, 16>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    if (f < nf) {
      let k = ctx.fbase + FW * f;
      for (var j = 0u; j < FW; j++) { w[j] = scratch[k + j]; }
      alive = select(0u, 1u, (w[0] & ALIVE) != 0u);
    }
    let s = scan_u(alive, lid);   // includes barriers: reads above complete before writes below
    if (alive == 1u) {
      let k = ctx.fbase + FW * (run + s.prefix);
      for (var j = 0u; j < FW; j++) { scratch[k + j] = w[j]; }
    }
    run += s.total;
    workgroupBarrier();
  }
  if (lid == 0u) { h_nf = run; h_compacted = 1u; }
  workgroupBarrier();
}

// Insert point pIdx into the hull (all threads). The visible faces are
// restricted to the edge-connected component around the most visible face
// (with exact predicates the visible set is connected; the walk is kept as
// a guard), the horizon is the boundary of that component.
fn hull_insert(ctx: HullCtx, pIdx: u32, lid: u32) {
  let p = qvert(ctx, pIdx);
  // Compact before collecting visible slots: compaction renumbers slots, so it
  // must never run between the visibility pass and the horizon pass.
  var nf = workgroupUniformLoad(&h_nf);
  let nalive = workgroupUniformLoad(&h_nalive);
  if (nf + 3u * VIS_MAX > ctx.fmax || nf > 2u * nalive + 512u) {
    hull_compact(ctx, lid);
    nf = workgroupUniformLoad(&h_nf);
  }
  if (lid == 0u) { atomicStore(&h_nvis, 0u); }
  for (var i = lid; i < VIS_MAX / 32u; i += 256u) { atomicStore(&h_conn[i], 0u); }
  workgroupBarrier();
  var bestD = -1e30; var bestK = 0u;
  for (var f = lid; f < nf; f += 256u) {
    let dd = face_eval(ctx, f, p);
    if (dd > 0.0) {
      let k = atomicAdd(&h_nvis, 1u);
      if (k < VIS_MAX) { h_vis[k] = f; if (dd > bestD) { bestD = dd; bestK = k; } }
    }
  }
  workgroupBarrier();
  if (lid == 0u) { h_tmp = atomicLoad(&h_nvis); }
  let nvis = workgroupUniformLoad(&h_tmp);
  if (nvis == 0u) { return; }
  if (nvis > VIS_MAX) {
    if (lid == 0u) { h_fail = 1u; }
    workgroupBarrier();
    return;
  }
  // seed = most visible face; grow the connected component by label propagation
  let seedK = argmax_f(bestD, bestK, lid);
  if (lid == 0u) { atomicOr(&h_conn[seedK >> 5u], 1u << (seedK & 31u)); }
  workgroupBarrier();
  loop {
    if (lid == 0u) { atomicStore(&h_changed, 0u); }
    workgroupBarrier();
    for (var i = lid; i < nvis; i += 256u) {
      if (conn_get(i)) { continue; }
      let fi = face_a(ctx, h_vis[i]);
      var hit = false;
      for (var j = 0u; j < nvis; j++) {
        if (j != i && conn_get(j) && shares_edge(fi, face_a(ctx, h_vis[j]))) { hit = true; break; }
      }
      if (hit) { atomicOr(&h_conn[i >> 5u], 1u << (i & 31u)); atomicStore(&h_changed, 1u); }
    }
    workgroupBarrier();
    if (lid == 0u) { h_tmp = atomicLoad(&h_changed); }
    let ch = workgroupUniformLoad(&h_tmp);
    if (ch == 0u) { break; }
  }
  if (nf + 3u * nvis > ctx.fmax) {
    if (lid == 0u) { h_fail = 2u; }
    workgroupBarrier();
    return;
  }
  if (lid == 0u) { atomicStore(&h_nnew, 0u); }
  workgroupBarrier();
  for (var e = lid; e < nvis * 3u; e += 256u) {
    let vi = e / 3u; let k = e % 3u;
    if (!conn_get(vi)) { continue; }
    let f = face_a(ctx, h_vis[vi]);
    let a = select(select(f.z, f.y, k == 1u), f.x, k == 0u);
    let b = select(select(f.x, f.z, k == 1u), f.y, k == 0u);
    var found = false;
    for (var j = 0u; j < nvis; j++) {
      if (!conn_get(j)) { continue; }
      let g = face_a(ctx, h_vis[j]);
      if ((g.x == b && g.y == a) || (g.y == b && g.z == a) || (g.z == b && g.x == a)) { found = true; break; }
    }
    if (!found) {
      let slot = nf + atomicAdd(&h_nnew, 1u);
      face_write(ctx, slot, a, b, pIdx);
    }
  }
  var killed = 0u;
  for (var vi = lid; vi < nvis; vi += 256u) { if (conn_get(vi)) { face_kill(ctx, h_vis[vi]); killed += 1u; } }
  let rKilled = reduce_u(killed, lid);
  if (lid == 0u) { let nn = atomicLoad(&h_nnew); h_nf = nf + nn; h_nalive = h_nalive + nn - rKilled; }
  workgroupBarrier();
}

// Farthest-first insertion of the candidate list (point indices in
// scratch[candBase..candBase+ncand)) until no candidate is outside the hull
// or h_fail is set. Returns the number of insertions. Uniform control flow.
fn hull_grow(ctx: HullCtx, ownerBase: u32, distBase: u32, candBase: u32, ncandIn: u32, nfPrevIn: u32, maxIter: u32, lid: u32) -> u32 {
  var ncand = ncandIn;
  var nfPrev = nfPrevIn;
  var sinceCompact = 0u;
  var nIns = 0u;
  for (var iter = 0u; iter < maxIter; iter++) {
    if (ncand == 0u) { break; }
    // farthest outstanding candidate
    var md = -1.0; var mi = NONE;
    for (var k = lid; k < ncand; k += 256u) {
      let i = scratch[candBase + k];
      if (scratch[ownerBase + i] != NONE) {
        let dd = bitcast<f32>(scratch[distBase + i]);
        if (dd > md) { md = dd; mi = i; }
      }
    }
    let pick = argmax_f(md, select(0u, mi, mi != NONE), lid);
    let pickD = workgroupUniformLoad(&sh_f1);
    if (pickD < 0.0) { break; }
    if (lid == 0u) { h_compacted = 0u; }
    workgroupBarrier();
    hull_insert(ctx, pick, lid);
    nIns += 1u;
    let fail = workgroupUniformLoad(&h_fail);
    if (fail != 0u) { break; }
    let nfNow = workgroupUniformLoad(&h_nf);
    let compacted = workgroupUniformLoad(&h_compacted);
    if (lid == 0u) { scratch[ownerBase + pick] = NONE; scratch[distBase + pick] = 0u; }
    // redistribute candidates whose owner died (or all after a face compaction)
    for (var k = lid; k < ncand; k += 256u) {
      let i = scratch[candBase + k];
      let own = scratch[ownerBase + i];
      if (own == NONE) { continue; }
      if (compacted == 0u && face_alive(ctx, own)) { continue; }
      let pt = qvert(ctx, i);
      var bestF = NONE; var bestD = -1.0;
      let f0 = select(nfPrev, 0u, compacted == 1u);
      for (var f = f0; f < nfNow; f++) {
        let dd = face_eval(ctx, f, pt);
        if (dd > 0.0 && dd > bestD) { bestD = dd; bestF = f; }
      }
      scratch[ownerBase + i] = bestF;
      scratch[distBase + i] = bitcast<u32>(select(0.0, bestD, bestF != NONE));
    }
    storageBarrier();
    workgroupBarrier();
    nfPrev = nfNow;
    // periodically drop interior points from the candidate list
    sinceCompact += 1u;
    if (sinceCompact >= 4u) {
      sinceCompact = 0u;
      var nc = 0u;
      let cChunks = (ncand + 255u) / 256u;
      for (var c = 0u; c < cChunks; c++) {
        let k = c * 256u + lid;
        var keep = 0u; var i = 0u;
        if (k < ncand) { i = scratch[candBase + k]; keep = select(0u, 1u, scratch[ownerBase + i] != NONE); }
        let sc = scan_u(keep, lid);   // barriers inside: reads done before writes
        if (keep == 1u) { scratch[candBase + nc + sc.prefix] = i; }
        nc += sc.total;
        workgroupBarrier();
      }
      if (lid == 0u) { h_ncand = nc; }
      ncand = workgroupUniformLoad(&h_ncand);
      storageBarrier();
      workgroupBarrier();
    }
  }
  return nIns;
}

// volume of the current hull (alive faces) in grid units; uniform control flow
fn hull_volume(ctx: HullCtx, o: vec3<f32>, lid: u32) -> f32 {
  let nf = workgroupUniformLoad(&h_nf);
  var vol = 0.0;
  for (var f = lid; f < nf; f += 256u) {
    if (face_alive(ctx, f)) {
      let fa = face_a(ctx, f);
      vol += tri_flux(qvert(ctx, fa.x), qvert(ctx, fa.y), qvert(ctx, fa.z), o);
    }
  }
  return abs(reduce_f(vol, lid));
}

@compute @workgroup_size(256)
fn hull(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
  let lid = lid3.x;
  let g = wg_linear(wg);
  if (g >= P.hullChunkSize) { return; }
  let partIdx = P.partBase + P.hullChunkBase + g;
  if (P.hullChunkBase + g >= P.nChildParts) { return; }
  if (lid == 0u) {
    let pd = parts[partIdx];
    sh_u1 = select(0u, 1u, pd.nv >= 4u && (pd.nt > 0u || (pd.flags & 2u) != 0u));
    sh_dims = vec4<u32>(pd.nv, pd.nt, pd.vOff, 0u);
    h_fail = 0u;
    h_meshVol = pd.meshVol;
  }
  let ok = workgroupUniformLoad(&sh_u1);
  let dims = workgroupUniformLoad(&sh_dims);
  if (ok == 0u) {
    if (lid == 0u) { parts[partIdx].hullVol = 0.0; }
    return;
  }
  var part: PartDesc;
  part.nv = dims.x; part.nt = dims.y; part.vOff = dims.z;
  var ctx: HullCtx;
  ctx.slab = P.hullSlabBase + g * P.hullSlabWords;
  ctx.vOff = part.vOff;
  ctx.nv = part.nv;
  let fb = (2u * part.nv + max(part.nv, 80u) + 7u) & ~7u;   // owner[nv], dist[nv], cand[max(nv, 80)]
  ctx.fbase = ctx.slab + fb;
  ctx.fmax = select(0u, (P.hullSlabWords - fb) / FW, P.hullSlabWords > fb);
  if (ctx.fmax < 64u) {
    if (lid == 0u) { parts[partIdx].hullVol = -1.0; parts[partIdx].flags = 3u; atomicAdd(&ctr[CTR_HULLFAIL], 1u); }
    return;
  }

  // bbox -> quantization grid (2^21 cells across the largest extent)
  var lo = vec3<f32>(1e30); var hi = vec3<f32>(-1e30);
  for (var i = lid; i < part.nv; i += 256u) { let p = vert(part.vOff, i); lo = min(lo, p); hi = max(hi, p); }
  bbox_reduce(lo, hi, lid);
  let blo = workgroupUniformLoad(&sh_v1);
  let bhi = workgroupUniformLoad(&sh_v2);
  let ext = bhi - blo;
  let scale = max(max(max(ext.x, ext.y), ext.z), 1e-30);
  ctx.qlo = blo;
  ctx.qs = 2097152.0 / scale;
  let cell = scale / 2097152.0;
  let volScale = cell * cell * cell;

  // extremes along 40 icosphere directions (80 seeds): their hull is a tight
  // inner approximation, so few points survive the first assignment pass.
  // Two axes per pass over the points, four argmax reductions at once (max
  // and min per axis); h_vis holds the indices while it is otherwise idle.
  for (var ps = 0u; ps < 20u; ps++) {
    let a0 = KDOP_AXES[2u * ps]; let a1 = KDOP_AXES[2u * ps + 1u];
    var mx0 = -1e30; var mx1 = -1e30; var mn0 = 1e30; var mn1 = 1e30;
    var ix0 = 0u; var ix1 = 0u; var in0 = 0u; var in1 = 0u;
    for (var i = lid; i < part.nv; i += 256u) {
      let v = qvert(ctx, i);
      let d0 = dot(a0, v); let d1 = dot(a1, v);
      if (d0 > mx0) { mx0 = d0; ix0 = i; }  if (d0 < mn0) { mn0 = d0; in0 = i; }
      if (d1 > mx1) { mx1 = d1; ix1 = i; }  if (d1 < mn1) { mn1 = d1; in1 = i; }
    }
    // rows 2k = max of axis k, 2k+1 = min (stored negated so every row is an argmax)
    h_red_f[lid] = mx0; h_vis[lid] = ix0; h_red_f[256u + lid] = -mn0; h_vis[256u + lid] = in0;
    h_red_f[512u + lid] = mx1; h_vis[512u + lid] = ix1; h_red_f[768u + lid] = -mn1; h_vis[768u + lid] = in1;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st = st >> 1u) {
      if (lid < st) {
        for (var r = 0u; r < 4u; r++) {
          let b0 = r * 256u + lid;
          let a = h_red_f[b0]; let b = h_red_f[b0 + st];
          let ia = h_vis[b0]; let ib = h_vis[b0 + st];
          if (b > a || (b == a && ib < ia)) { h_red_f[b0] = b; h_vis[b0] = ib; }
        }
      }
      workgroupBarrier();
    }
    if (lid < 4u) { h_seed[4u * ps + lid] = h_vis[lid * 256u]; }
    workgroupBarrier();
  }
  if (P.pad4 == 1u) { if (lid == 0u) { parts[partIdx].hullVol = 0.0; } return; }

  // initial simplex from the seeds (lid 0 evaluates the 80 seeds serially,
  // exact tests decide degeneracy)
  if (lid == 0u) {
    let i0 = h_seed[1]; let i1 = h_seed[0];
    let p0 = qvert(ctx, i0); let p1 = qvert(ctx, i1);
    var i2 = NONE; var best = 0.0;
    let d01 = p1 - p0;
    for (var k = 0u; k < 80u; k++) {
      let c = cross(d01, qvert(ctx, h_seed[k]) - p0);
      let v = dot(c, c);
      if (v > best) { best = v; i2 = h_seed[k]; }
    }
    var i3 = NONE;
    if (i2 != NONE) {
      let p2 = qvert(ctx, i2);
      let e2 = p2 - p0;
      let n = cross(d01, e2);
      var bd = 0.0; var bs = 0;
      for (var k = 0u; k < 80u; k++) {
        let dk = qvert(ctx, h_seed[k]) - p0;
        let v = abs(dot(n, dk));
        if (v > bd) {
          let sgn = orient_filtered(n, d01, e2, dk);
          if (sgn != 0) { bd = v; i3 = h_seed[k]; bs = sgn; }
        }
      }
      if (i3 != NONE) {
        var a = i0; var b = i1; var c = i2;
        if (bs > 0) { b = i2; c = i1; }
        face_write(ctx, 0u, a, b, c);
        face_write(ctx, 1u, a, i3, b);
        face_write(ctx, 2u, b, i3, c);
        face_write(ctx, 3u, c, i3, a);
        h_nf = 4u; h_nalive = 4u;
        sh_v1 = (qvert(ctx, a) + qvert(ctx, b) + qvert(ctx, c) + qvert(ctx, i3)) * 0.25;
        h_tmp = 1u;
      } else { h_tmp = 0u; }
    } else { h_tmp = 0u; }
  }
  let simplexOk = workgroupUniformLoad(&h_tmp);
  if (simplexOk == 0u) {
    if (lid == 0u) { parts[partIdx].hullVol = 0.0; }
    return;
  }
  let o = workgroupUniformLoad(&sh_v1);

  // Quickhull proper, in two rounds: first over the 80 seeds (their hull is
  // a tight inner approximation), then over every point still outside it.
  // Every insertion takes the globally farthest candidate.
  let ownerBase = ctx.slab;
  let distBase = ctx.slab + part.nv;
  let candBase = ctx.slab + 2u * part.nv;
  if (lid < 80u) {
    let i = h_seed[lid];
    let pt = qvert(ctx, i);
    var bestF = NONE; var bestD = -1.0;
    for (var f = 0u; f < 4u; f++) {
      let dd = face_eval(ctx, f, pt);
      if (dd > 0.0 && dd > bestD) { bestD = dd; bestF = f; }
    }
    scratch[ownerBase + i] = bestF;
    scratch[distBase + i] = bitcast<u32>(select(0.0, bestD, bestF != NONE));
    scratch[candBase + lid] = i;
  }
  storageBarrier();
  workgroupBarrier();
  var nIns = hull_grow(ctx, ownerBase, distBase, candBase, 80u, 4u, 80u, lid);
  var fail0 = workgroupUniformLoad(&h_fail);
  var nfPrev = workgroupUniformLoad(&h_nf);
  // Branch-and-bound at the last tree level (port of la_hull's rough-hull
  // pruning): the rough hull is inside the exact one, so its rv is a lower
  // bound on this child's level cost. If the parent path's partial sum plus
  // that bound already exceeds the best completed-path upper bound of the
  // seed (parent-level items, assuming no level gets worse than the current
  // worst part), the exact hull cannot matter: record the bound and stop.
  if (P.lastLevel == 1u && P.level > 0u && P.nSeeds > 0u && P.exportStride == 0u && fail0 == 0u) {
    if (lid == 0u) {
      let job = (P.hullChunkBase + g) / 2u;
      let jb = jobs[P.jobBase + job];
      let pit = items[P.itemsInBase + jb.parentItem];
      var ssum = 0.0;
      for (var l = 0u; l < pit.nLevels; l++) { ssum += pit.levelCosts[l]; }
      sh_dims = vec4<u32>(pit.srcPart, pit.nLevels, 0u, 0u);
      sh_f1 = ssum;
    }
    let pinfo = workgroupUniformLoad(&sh_dims);
    let partial = workgroupUniformLoad(&sh_f1);
    let perSeed = P.nItemsIn / P.nSeeds;
    var ubMin = 1e30;
    for (var i = lid; i < perSeed; i += 256u) {
      let it = items[P.itemsInBase + pinfo.x * perSeed + i];
      if (it.flags == 1u && it.nLevels > 0u) {
        var ps = 0.0;
        for (var l = 0u; l < it.nLevels; l++) { ps += it.levelCosts[l]; }
        let ub = ps + it.levelCosts[it.nLevels - 1u] * f32(P.totalLevels - it.nLevels);
        ubMin = min(ubMin, ub);
      }
    }
    let ub = reduce_min_f(ubMin, lid);
    let vRough = hull_volume(ctx, o, lid) * volScale;
    if (lid == 0u) { h_bound = vec2<f32>(partial + rv_cost(h_meshVol, vRough), ub); }
    let bnd = workgroupUniformLoad(&h_bound);
    if (bnd.x > bnd.y) {
      if (lid == 0u) { parts[partIdx].hullVol = vRough; parts[partIdx].flags = parts[partIdx].flags | 4u; atomicAdd(&ctr[CTR_PRUNED], 1u); }
      return;
    }
  }
  // assignment of every point against the rough hull; candidates = points outside
  var ncand = 0u;
  let nvChunks = (part.nv + 255u) / 256u;
  for (var c = 0u; c < nvChunks; c++) {
    let i = c * 256u + lid;
    var outside = 0u;
    if (i < part.nv && fail0 == 0u) {
      let pt = qvert(ctx, i);
      var bestF = NONE; var bestD = -1.0;
      for (var f = 0u; f < nfPrev; f++) {
        let dd = face_eval(ctx, f, pt);
        if (dd > 0.0 && dd > bestD) { bestD = dd; bestF = f; }
      }
      scratch[ownerBase + i] = bestF;
      scratch[distBase + i] = bitcast<u32>(select(0.0, bestD, bestF != NONE));
      outside = select(0u, 1u, bestF != NONE);
    }
    let sc = scan_u(outside, lid);
    if (outside == 1u) { scratch[candBase + ncand + sc.prefix] = i; }
    ncand += sc.total;
    workgroupBarrier();
  }
  if (lid == 0u) { h_ncand = ncand; if (P.exportStride == 0u) { parts[partIdx].pad0 = ncand; } }
  ncand = workgroupUniformLoad(&h_ncand);
  storageBarrier();
  workgroupBarrier();
  let maxIter = select(part.nv, 0u, P.pad4 == 2u || P.pad4 == 3u || fail0 != 0u);
  nIns += hull_grow(ctx, ownerBase, distBase, candBase, ncand, nfPrev, maxIter, lid);
  let fail = workgroupUniformLoad(&h_fail);
  if (fail != 0u) {
    if (lid == 0u) { parts[partIdx].hullVol = -1.0; parts[partIdx].flags = fail; atomicAdd(&ctr[CTR_HULLFAIL], 1u); }
    return;
  }
  // volume over alive faces (local grid units, then scaled back)
  let total = hull_volume(ctx, o, lid);
  let nf = workgroupUniformLoad(&h_nf);
  if (lid == 0u) { parts[partIdx].hullVol = total * volScale; parts[partIdx].pad1 = nIns; }
  // optional export of the alive faces (hull mesh) into the triangle pool
  if (P.exportStride != 0u) {
    var run = 0u;
    let nChunks = (nf + 255u) / 256u;
    let base = P.tBase + P.exportBase + (P.hullChunkBase + g) * P.exportStride;   // global index: chunks share the export region
    for (var c = 0u; c < nChunks; c++) {
      let f = c * 256u + lid;
      var alive = 0u;
      var fa = vec3<u32>(0u);
      if (f < nf && face_alive(ctx, f)) { alive = 1u; fa = face_a(ctx, f); }
      let sc = scan_u(alive, lid);
      if (alive == 1u && 3u * (run + sc.prefix) + 3u <= P.exportStride) {
        let k = base + 3u * (run + sc.prefix);
        pool[k] = fa.x; pool[k + 1u] = fa.y; pool[k + 2u] = fa.z;
      }
      run += sc.total;
      workgroupBarrier();
    }
    if (lid == 0u) {
      parts[partIdx].pad0 = min(run, P.exportStride / 3u);
      if (P.nJobs != 0u) {
        // descriptor of the hull as a mesh (verts of the part, exported faces),
        // read by a Hausdorff pass encoded in the same submission
        var hd: PartDesc;
        hd.vOff = part.vOff; hd.nv = part.nv; hd.tOff = P.exportBase + (P.hullChunkBase + g) * P.exportStride; hd.nt = min(run, P.exportStride / 3u);
        hd.eOff = 0u; hd.ne = 0u; hd.meshVol = 0.0; hd.hullVol = total * volScale;
        hd.planeIds = vec4<u32>(NONE); hd.nPlanes = 0u; hd.flags = 2u; hd.pad0 = 0u; hd.pad1 = 0u;
        parts[partIdx + P.nJobs] = hd;
      }
    }
  }
}
`;
// ---- evaluate: best first cut per seed, refinement bookkeeping ----------
// One workgroup per seed. Mirrors la_evaluate: the path cost of a leaf item
// (full-depth leaves at the last level, "extra leaves" that could not be
// cut further at earlier levels) is the mean of its level costs; the best
// over all leaves of a seed picks its first cut (lowest cut index on ties),
// with a greedy fallback on level-0 costs when no leaf is valid.
//
// State per seed lives in the planes buffer at P.hullChunkBase (vec4 units):
//   [0] tree header (depth, nSeeds, w0t, nProbes) as u32 bits, [1..4] per level (itemsOutBase, nJobs, 0, 0)
//   [5..9] the same for the probe-round layout (P.nChildParts selects the header: 0 or 5)
//   [16 + 4 s + 0] chosen plane
//   [16 + 4 s + 1] (bestCost, half-width of the probe scan, bestCut bits, flags bits: 1 = valid)
//   [16 + 4 s + 2] (initial half-width from the CPU, 0, 0, 0)
// Mode (P.hullSlabWords): 0 = after the tree, 1 = after a probe round (accept
// a probe only when it beats the current cost). Both modes then write the
// next round's probe planes into the seed's edge-plane slots (zero planes
// when the seed has no plane or no room to move).
const EVAL = /* wgsl */ `
var<workgroup> ev_f: array<f32, 256>;
var<workgroup> ev_u: array<u32, 256>;
var<workgroup> ev_found: u32;
fn ev_u4(i: u32) -> vec4<u32> { return bitcast<vec4<u32>>(planes[i]); }
// (min cost, lowest cut) reduction
fn ev_argmin(c: f32, cut: u32, lid: u32) -> vec2<u32> {
  ev_f[lid] = c; ev_u[lid] = cut;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st = st >> 1u) {
    if (lid < st) {
      let a = ev_f[lid]; let b = ev_f[lid + st];
      let ia = ev_u[lid]; let ib = ev_u[lid + st];
      if (b < a || (b == a && ib < ia)) { ev_f[lid] = b; ev_u[lid] = ib; }
    }
    workgroupBarrier();
  }
  let r = vec2<u32>(ev_u[0], bitcast<u32>(ev_f[0]));
  workgroupBarrier();
  return r;
}
@compute @workgroup_size(256)
fn evaluate(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
  let lid = lid3.x;
  let s = wg.x;
  if (s >= P.nSeeds) { return; }
  let sb = P.hullChunkBase;
  let hb = sb + P.nChildParts;   // header of this run's level layout (tree or probe round)
  let hdr = ev_u4(hb);
  let depth = hdr.x; let w0t = hdr.z; let nProbes = hdr.w;
  var bestC = 1e30; var bestCut = NONE;
  for (var l = 0u; l < depth; l++) {
    let lh = ev_u4(hb + 1u + l);
    let perSeed = lh.y / P.nSeeds;
    let wantFlag = select(2u, 1u, l == depth - 1u);
    for (var i = lid; i < perSeed; i += 256u) {
      let it = items[lh.x + s * perSeed + i];
      if (it.flags != wantFlag || it.nLevels == 0u || it.initialCut == NONE) { continue; }
      var sum = 0.0;
      for (var k = 0u; k < it.nLevels; k++) { sum += it.levelCosts[k]; }
      let c = sum / f32(it.nLevels);
      if (c < bestC || (c == bestC && it.initialCut < bestCut)) { bestC = c; bestCut = it.initialCut; }
    }
  }
  var r = ev_argmin(bestC, bestCut, lid);
  if (lid == 0u) { ev_found = select(0u, 1u, r.x != NONE); }
  let found = workgroupUniformLoad(&ev_found);
  if (found == 0u) {
    // greedy fallback on the level-0 costs
    let lh = ev_u4(hb + 1u);
    let perSeed = lh.y / P.nSeeds;
    bestC = 1e30; bestCut = NONE;
    for (var i = lid; i < perSeed; i += 256u) {
      let it = items[lh.x + s * perSeed + i];
      if (it.flags == 1u && it.nLevels >= 1u && it.initialCut != NONE) {
        let c = it.levelCosts[0];
        if (c < bestC || (c == bestC && it.initialCut < bestCut)) { bestC = c; bestCut = it.initialCut; }
      }
    }
    r = ev_argmin(bestC, bestCut, lid);
  }
  if (lid != 0u) { return; }
  let cut = r.x;
  let cost = bitcast<f32>(r.y);
  let st = sb + 16u + 4u * s;
  var plane = planes[st];
  var s1 = planes[st + 1u];
  var half = s1.y;
  var valid = (bitcast<u32>(s1.w) & 1u) != 0u;
  if (P.hullSlabWords == 0u) {
    // after the tree: take the best cut as is
    valid = cut != NONE;
    if (valid) { plane = planes[P.jobBase + s * w0t + cut]; }
    s1 = vec4<f32>(select(1e30, cost, valid), half, bitcast<f32>(cut), bitcast<f32>(select(0u, 1u, valid)));
    half = planes[st + 2u].x;
  } else {
    // after a probe round: accept an improvement only
    if (cut != NONE && cost < s1.x - 1e-7) {
      plane = planes[P.jobBase + s * w0t + cut];
      s1 = vec4<f32>(cost, s1.y, bitcast<f32>(cut), s1.w);
    }
    half = select(0.0, 2.0 * half / f32(nProbes - 1u), nProbes > 1u);
  }
  s1.y = half;
  planes[st] = plane;
  planes[st + 1u] = s1;
  // probe planes for the next round
  for (var i = 0u; i < nProbes; i++) {
    var pp = vec4<f32>(0.0);
    if (valid && half > 0.0 && nProbes > 1u) {
      let off = half * ((2.0 * f32(i)) / f32(nProbes - 1u) - 1.0);
      pp = vec4<f32>(plane.xyz, plane.w - off);
    }
    planes[P.edgePlaneBase + s * P.edgeStride + i] = pp;
  }
}
`;
const RECORD = /* wgsl */ `
@compute @workgroup_size(256)
fn record(@builtin(global_invocation_id) gid: vec3<u32>) {
  let g = gid.x;
  if (g >= P.nJobs) { return; }
  var it = items[P.itemsOutBase + g];
  if (it.flags != 1u) { return; }
  var costs: array<f32, 8>;
  var bad = false;
  for (var i = 0u; i < it.np; i++) {
    let pd = parts[it.partIds[i]];
    if (pd.hullVol < 0.0 || (pd.flags & 4u) != 0u) { bad = true; }
    costs[i] = rv_cost(pd.meshVol, pd.hullVol);
  }
  if (bad) { it.flags = 0u; items[P.itemsOutBase + g] = it; return; }
  // insertion sort ascending by rv cost
  for (var i = 1u; i < it.np; i++) {
    let kc = costs[i]; let kp = it.partIds[i];
    var j = i;
    while (j > 0u && costs[j - 1u] > kc) { costs[j] = costs[j - 1u]; it.partIds[j] = it.partIds[j - 1u]; j -= 1u; }
    costs[j] = kc; it.partIds[j] = kp;
  }
  if (it.nLevels < 4u) { it.levelCosts[it.nLevels] = costs[it.np - 1u]; it.nLevels += 1u; }
  items[P.itemsOutBase + g] = it;
}
`;
export const SHADER_SOURCE = COMMON + PLAN + SCAN + EMIT + HULL + EVAL + RECORD + HAUSDORFF_SHADER;
// Host mirrors of the struct layouts (in u32 words).
export const PART_WORDS = 16;
export const ITEM_WORDS = 20;
export const JOB_WORDS = 24;
export const PARAMS_WORDS = 32;
