// hausdorff_shader.ts — GPU bidirectional sampled Hausdorff distance between
// pairs of meshes (part vs. its hull), mirroring cuda/hausdorff.cuh's sampling
// rule (area-proportional, Wang-hash barycentrics) with brute-force nearest
// triangle search. One workgroup per pair; both meshes are uploaded as seed
// parts (A at index k, B at index n + k). Sample scratch per pair is given by
// parts[k].eOff (offset in scratch words) and parts[k].ne (capacity in
// samples); the result is written to parts[k].pad1 as a bitcast f32.
//
// Shares the bind group layout of the tree kernels (the string is appended to
// the same module).

export const HAUSDORFF_SHADER = /* wgsl */ `
const HD_RESOLUTION = 2000.0;
const HD_MIN_SAMPLES = 1000.0;

fn hd_wang(seedIn: u32) -> u32 {
  var seed = (seedIn ^ 61u) ^ (seedIn >> 16u);
  seed = seed * 9u;
  seed = seed ^ (seed >> 4u);
  seed = seed * 0x27d4eb2du;
  seed = seed ^ (seed >> 15u);
  return seed;
}
fn hd_tri_area(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> f32 { return 0.5 * length(cross(b - a, c - a)); }
fn hd_seg_sq(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>) -> f32 {
  let ab = b - a; let ap = p - a;
  let ab2 = dot(ab, ab);
  var t = select(0.0, dot(ap, ab) / ab2, ab2 > 1e-20);
  t = clamp(t, 0.0, 1.0);
  let d = ap - t * ab;
  return dot(d, d);
}
fn hd_tri_sq(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> f32 {
  let ab = b - a; let ac = c - a; let ap = p - a;
  let n = cross(ab, ac);
  let n2 = dot(n, n);
  if (n2 < 1e-20) {
    let bp = p - b; let cp = p - c;
    return min(dot(ap, ap), min(dot(bp, bp), dot(cp, cp)));
  }
  let d00 = dot(ab, ab); let d01 = dot(ab, ac); let d02 = dot(ab, ap);
  let d11 = dot(ac, ac); let d12 = dot(ac, ap);
  let inv = 1.0 / (d00 * d11 - d01 * d01);
  let u = (d11 * d02 - d01 * d12) * inv;
  let v = (d00 * d12 - d01 * d02) * inv;
  if (u >= 0.0 && v >= 0.0 && u + v <= 1.0) {
    let dist = dot(n, ap);
    return (dist * dist) / n2;
  }
  return min(hd_seg_sq(p, a, b), min(hd_seg_sq(p, b, c), hd_seg_sq(p, c, a)));
}

var<workgroup> hd_run: u32;
var<workgroup> hd_tot: u32;

// Sample mesh m into scratch[base .. base + 3*cap) — only sample indices in
// [sliceLo, sliceHi) are written; returns the total sample count.
fn hd_sample(vOff: u32, tOff: u32, nt: u32, base: u32, cap: u32, sliceLo: u32, sliceHi: u32, lid: u32) -> u32 {
  // total area
  var area = 0.0;
  for (var t = lid; t < nt; t += 256u) {
    let tr = tri(tOff, t);
    area += hd_tri_area(vert(vOff, tr.x), vert(vOff, tr.y), vert(vOff, tr.z));
  }
  let total = reduce_f(area, lid);
  var res = floor(HD_RESOLUTION * total);
  if (res < HD_MIN_SAMPLES) { res = HD_MIN_SAMPLES; }
  let step = select(0u, nt / u32(res), f32(nt) > res);
  // per-triangle counts, chunked exclusive scan, sample generation
  if (lid == 0u) { hd_run = 0u; }
  workgroupBarrier();
  let nChunks = (nt + 255u) / 256u;
  for (var c = 0u; c < nChunks; c++) {
    let t = c * 256u + lid;
    var N = 0u;
    var pa = vec3<f32>(0.0); var pb = pa; var pc = pa;
    if (t < nt) {
      let tr = tri(tOff, t);
      pa = vert(vOff, tr.x); pb = vert(vOff, tr.y); pc = vert(vOff, tr.z);
      let ta = hd_tri_area(pa, pb, pc);
      let areaCount = select(0u, u32(floor(res / total * ta)), total > 1e-20);
      if (f32(nt) > res) { N = select(select(0u, 1u, (t % max(step, 1u)) == 0u), areaCount, areaCount > 0u); }
      else { N = select(select(0u, 1u, (t % 2u) == 0u), areaCount, areaCount > 0u); }
    }
    let baseRun = hd_run;
    let sc = scan_u(N, lid);
    let off = baseRun + sc.prefix;
    for (var k = 0u; k < N; k++) {
      let idx = off + k;
      if (idx < cap && idx >= sliceLo && idx < sliceHi) {
        let ha = hd_wang(t * 65537u + k);
        let hb = hd_wang(ha + 0x9e3779b9u);
        let ra = f32(ha & 0xFFFFFFu) / f32(0xFFFFFFu);
        let rb = f32(hb & 0xFFFFFFu) / f32(0xFFFFFFu);
        let sqa = sqrt(ra);
        let s = (1.0 - sqa) * pa + sqa * (1.0 - rb) * pb + sqa * rb * pc;
        let w = base + 3u * idx;
        scratch[w] = bitcast<u32>(s.x); scratch[w + 1u] = bitcast<u32>(s.y); scratch[w + 2u] = bitcast<u32>(s.z);
      }
    }
    if (lid == 255u) { hd_run = baseRun + sc.total; }
    workgroupBarrier();
  }
  if (lid == 0u) { hd_tot = min(hd_run, cap); }
  let n = workgroupUniformLoad(&hd_tot);
  storageBarrier();
  workgroupBarrier();
  return n;
}

// max over samples [lo, hi) in scratch[base..) of the squared distance to mesh m
fn hd_directed_sq(base: u32, lo: u32, hi: u32, vOff: u32, tOff: u32, nt: u32, lid: u32) -> f32 {
  var worst = 0.0;
  for (var i = lo + lid; i < hi; i += 256u) {
    let w = base + 3u * i;
    let p = vec3<f32>(bitcast<f32>(scratch[w]), bitcast<f32>(scratch[w + 1u]), bitcast<f32>(scratch[w + 2u]));
    var best = 1e30;
    for (var t = 0u; t < nt; t++) {
      let tr = tri(tOff, t);
      let d = hd_tri_sq(p, vert(vOff, tr.x), vert(vOff, tr.y), vert(vOff, tr.z));
      best = min(best, d);
    }
    worst = max(worst, best);
  }
  return reduce_max_f(worst, lid);
}

@compute @workgroup_size(256)
fn hausdorff(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
  let lid = lid3.x;
  let g = wg_linear(wg);
  // P.nJobs pairs, P.hullChunkSize workgroups per pair (sample slices of 256)
  let pair = g / P.hullChunkSize;
  let slice = g % P.hullChunkSize;
  if (pair >= P.nJobs) { return; }
  let a = parts[pair];
  let b = parts[P.nJobs + pair];
  if (lid == 0u) { sh_dims = vec4<u32>(a.nt, b.nt, a.eOff, a.ne); }
  let dims = workgroupUniformLoad(&sh_dims);
  let ntA = dims.x; let ntB = dims.y;
  let base = dims.z; let cap = dims.w;
  if (ntA == 0u || ntB == 0u) { return; }
  let capA = cap / 2u; let capB = cap - capA;
  // slices index the concatenated sample range [0, capA + capB)
  let lo = slice * 256u; let hi = lo + 256u;
  let nsA = hd_sample(a.vOff, a.tOff, ntA, base, capA, lo, hi, lid);
  let nsB = hd_sample(b.vOff, b.tOff, ntB, base + 3u * capA, capB, select(0u, lo - capA, lo > capA), select(0u, hi - capA, hi > capA), lid);
  var worst = 0.0;
  if (lo < nsA) { worst = max(worst, hd_directed_sq(base, lo, min(hi, nsA), b.vOff, b.tOff, ntB, lid)); }
  let loB = select(0u, lo - capA, lo > capA); let hiB = select(0u, hi - capA, hi > capA);
  if (hiB > loB && loB < nsB) { worst = max(worst, hd_directed_sq(base + 3u * capA, loB, min(hiB, nsB), a.vOff, a.tOff, ntA, lid)); }
  if (lid == 0u) { atomicMax(&ctr[8u + pair], bitcast<u32>(sqrt(worst))); }
}
`;
