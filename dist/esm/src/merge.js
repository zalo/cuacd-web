// merge.ts — greedy hull-merging post-pass (port of CoACD MergeConvexHulls /
// cuda/postprocess_merge.cu).
//
// For every pair of parts that survives two cheap rejections (rv of the two
// hulls' combined gap, bbox distance), the hull of the union is built on the
// GPU and the pair cost is max(rv, Hausdorff) of the merged hull against the
// union of the two meshes. The Hausdorff ignores triangles lying on the seam:
// the face plane of A's hull that separates B and carries the most coplanar
// mesh area (CoACD's ComputeOverlapFace picks the first separating face), plus
// the seams of earlier merges recorded on the parts. Pairs under the threshold
// are merged by greedy min-cost matching; the pass repeats, re-evaluating only
// pairs that involve a new part, until no pair qualifies. One round is what
// CuACD does; the repeated rounds approximate CoACD's sequential loop.
//
// The merged mesh is the plain concatenation (both caps stay inside), as in
// CoACD and CuACD; it is still closed in the edge-twin sense.
import { bbox, mergeMeshes } from "./mesh.js";
import { cleanHullFaces, compactHull, convexHull } from "./cpu/hull.js";
import { hausdorff as cpuHausdorff } from "./cpu/hausdorff.js";
import { rvCost } from "./part.js";
export const DEFAULT_MERGE_OPTIONS = {
    threshold: 0.05,
    maxRounds: 32,
    planeTol: 1e-3,
    hullEps: 2e-5,
    hullBackend: "gpu",
    hausdorffBackend: "gpu",
    verbose: 0,
};
function bboxGap2(a, b) {
    let g2 = 0;
    for (let k = 0; k < 3; k++) {
        const g = Math.max(0, a.lo[k] - b.hi[k], b.lo[k] - a.hi[k]);
        g2 += g * g;
    }
    return g2;
}
function triCoplanar(V, T, t, p, tol) {
    for (let e = 0; e < 3; e++) {
        const i = T[t + e] * 3;
        if (Math.abs(p[0] * V[i] + p[1] * V[i + 1] + p[2] * V[i + 2] + p[3]) > tol)
            return false;
    }
    return true;
}
function coplanarArea(m, p, tol) {
    const V = m.verts, T = m.tris;
    let area = 0;
    for (let t = 0; t < T.length; t += 3) {
        if (!triCoplanar(V, T, t, p, tol))
            continue;
        const a = T[t] * 3, b = T[t + 1] * 3, c = T[t + 2] * 3;
        const abx = V[b] - V[a], aby = V[b + 1] - V[a + 1], abz = V[b + 2] - V[a + 2];
        const acx = V[c] - V[a], acy = V[c + 1] - V[a + 1], acz = V[c + 2] - V[a + 2];
        area += 0.5 * Math.hypot(aby * acz - abz * acy, abz * acx - abx * acz, abx * acy - aby * acx);
    }
    return area;
}
/**
 * Seam plane between two parts: a face plane of A's hull with all of B's hull
 * on its outer side (or on it), choosing the one with the largest coplanar
 * mesh area in A and B. Null when no separating face carries any mesh area.
 * Planes use the cut.ts convention (a, b, c, d) with a x + b y + c z + d = 0.
 */
export function seamPlane(a, b, tol) {
    const hv = a.hull.verts, ht = a.hull.tris, bv = b.hull.verts;
    const seen = new Set();
    let best = null, bestArea = 0;
    for (let t = 0; t < ht.length; t += 3) {
        const i0 = ht[t] * 3, i1 = ht[t + 1] * 3, i2 = ht[t + 2] * 3;
        const e1x = hv[i1] - hv[i0], e1y = hv[i1 + 1] - hv[i0 + 1], e1z = hv[i1 + 2] - hv[i0 + 2];
        const e2x = hv[i2] - hv[i0], e2y = hv[i2 + 1] - hv[i0 + 1], e2z = hv[i2 + 2] - hv[i0 + 2];
        let nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
        const len = Math.hypot(nx, ny, nz);
        if (len < 1e-20)
            continue;
        nx /= len;
        ny /= len;
        nz /= len;
        const d = nx * hv[i0] + ny * hv[i0 + 1] + nz * hv[i0 + 2];
        const key = `${Math.round(nx * 2000)},${Math.round(ny * 2000)},${Math.round(nz * 2000)},${Math.round(d * 2000)}`;
        if (seen.has(key))
            continue;
        seen.add(key);
        // B must lie on the outer side of the face (hull faces are wound outward)
        let ok = true;
        for (let j = 0; j < bv.length; j += 3) {
            if (nx * bv[j] + ny * bv[j + 1] + nz * bv[j + 2] - d < -tol) {
                ok = false;
                break;
            }
        }
        if (!ok)
            continue;
        const p = [nx, ny, nz, -d]; // cut.ts convention: n·x + d = 0
        const area = coplanarArea(a.mesh, p, tol) + coplanarArea(b.mesh, p, tol);
        if (area > bestArea) {
            bestArea = area;
            best = p;
        }
    }
    return best;
}
/** Concatenation of two meshes without the triangles lying on any of the given planes. */
export function filteredConcat(a, b, planes, tol) {
    const m = mergeMeshes(a, b);
    if (planes.length === 0)
        return m;
    const V = m.verts, T = m.tris;
    const out = new Uint32Array(T.length);
    let w = 0;
    for (let t = 0; t < T.length; t += 3) {
        let drop = false;
        for (const p of planes)
            if (triCoplanar(V, T, t, p, tol)) {
                drop = true;
                break;
            }
        if (drop)
            continue;
        out[w++] = T[t];
        out[w++] = T[t + 1];
        out[w++] = T[t + 2];
    }
    return { verts: V, tris: out.slice(0, w) };
}
function concatVerts(a, b) {
    const v = new Float32Array(a.length + b.length);
    v.set(a);
    v.set(b, a.length);
    return v;
}
/**
 * Merge parts whose union is convex enough (cost below the threshold).
 * Returns the new part list; the input array is not modified.
 */
export async function mergeHulls(tree, input, o = {}) {
    const opts = { ...DEFAULT_MERGE_OPTIONS, ...o };
    const t0 = performance.now();
    const stats = { rounds: 0, merges: 0, pairsHulled: 0, pairsHausdorff: 0, hullFallbacks: 0, hullMs: 0, hausdorffMs: 0, totalMs: 0 };
    let parts = input.slice();
    let nextId = 0;
    const ids = new Map();
    const boxes = new Map();
    const idOf = (p) => { let id = ids.get(p); if (id === undefined) {
        id = nextId++;
        ids.set(p, id);
        boxes.set(p, bbox(p.mesh));
    } return id; };
    const cache = new Map();
    const key = (a, b) => { const x = idOf(a), y = idOf(b); return x < y ? `${x}:${y}` : `${y}:${x}`; };
    const useGpuHull = opts.hullBackend === "gpu" && tree !== null;
    const useGpuHd = opts.hausdorffBackend === "gpu" && tree !== null;
    const halfT2 = (opts.threshold / 2) ** 2;
    for (let round = 0; round < opts.maxRounds && parts.length > 1; round++) {
        if (opts.signal?.aborted)
            throw new DOMException("merge aborted", "AbortError");
        stats.rounds = round + 1;
        // ---- candidate pairs after the cheap rejections ----
        const todo = [];
        const candidates = [];
        for (let i = 0; i < parts.length; i++) {
            for (let j = i + 1; j < parts.length; j++) {
                const A = parts[i], B = parts[j];
                const k = key(A, B);
                if (cache.has(k)) {
                    candidates.push([i, j]);
                    continue;
                }
                if (rvCost(A.meshVol + B.meshVol, A.hullVol + B.hullVol) >= opts.threshold)
                    continue;
                if (bboxGap2(boxes.get(A), boxes.get(B)) >= halfT2)
                    continue;
                todo.push([i, j]);
                candidates.push([i, j]);
            }
        }
        opts.onRound?.({ round, nParts: parts.length, candidates: candidates.length, merged: -1 });
        // ---- merged hulls (GPU, batched) ----
        const th0 = performance.now();
        const evals = new Array(todo.length);
        const pointSets = todo.map(([i, j]) => concatVerts(parts[i].hull.verts, parts[j].hull.verts));
        const hulls = new Array(todo.length).fill(null);
        if (useGpuHull) {
            const batch = 1024;
            for (let s = 0; s < todo.length; s += batch) {
                const chunk = pointSets.slice(s, s + batch);
                const res = await tree.hullMeshes(chunk.map((v) => ({ verts: v, tris: new Uint32Array(0) })));
                res.forEach((h, k) => { if (h)
                    hulls[s + k] = { mesh: compactHull(chunk[k], cleanHullFaces(chunk[k], h.tris, opts.hullEps)).mesh, volume: h.volume }; });
            }
        }
        for (let k = 0; k < todo.length; k++) {
            if (!hulls[k]) {
                hulls[k] = convexHull(pointSets[k]);
                stats.hullFallbacks++;
            }
            const [i, j] = todo[k];
            const h = hulls[k];
            const rv = rvCost(parts[i].meshVol + parts[j].meshVol, h.volume);
            evals[k] = { cost: rv, hull: rv < opts.threshold ? h.mesh : null, hullVol: h.volume, hd: 0, seam: null };
        }
        stats.hullMs += performance.now() - th0;
        stats.pairsHulled += todo.length;
        // ---- Hausdorff(filtered union, merged hull) for pairs still under the threshold ----
        const thd = performance.now();
        const hdIdx = [];
        const hdPairs = [];
        for (let k = 0; k < todo.length; k++) {
            const ev = evals[k];
            if (!ev.hull)
                continue;
            const [i, j] = todo[k];
            const A = parts[i], B = parts[j];
            ev.seam = seamPlane(A, B, opts.planeTol);
            const planes = [...(ev.seam ? [ev.seam] : []), ...(A.seams ?? []), ...(B.seams ?? [])];
            const m = filteredConcat(A.mesh, B.mesh, planes, opts.planeTol);
            if (m.tris.length === 0)
                continue;
            hdIdx.push(k);
            hdPairs.push({ a: m, b: ev.hull });
        }
        const hd = new Float32Array(hdPairs.length).fill(NaN);
        if (useGpuHd) {
            // as many pairs per call as the sample scratch allows (halve on overflow)
            let batch = 1024;
            for (let s = 0; s < hdPairs.length;) {
                try {
                    const r = await tree.hausdorff(hdPairs.slice(s, s + batch));
                    hd.set(r, s);
                    s += batch;
                }
                catch (e) {
                    if (batch > 1 && String(e.message).includes("scratch")) {
                        batch >>= 1;
                        continue;
                    }
                    throw e;
                }
            }
        }
        hdIdx.forEach((k, q) => {
            const v = Number.isFinite(hd[q]) ? hd[q] : cpuHausdorff(hdPairs[q].a, hdPairs[q].b);
            evals[k].hd = v;
            evals[k].cost = Math.max(evals[k].cost, v);
            if (evals[k].cost >= opts.threshold)
                evals[k].hull = null;
        });
        stats.hausdorffMs += performance.now() - thd;
        if (opts.onPair)
            todo.forEach(([i, j], k) => opts.onPair({ a: parts[i], b: parts[j], rv: rvCost(parts[i].meshVol + parts[j].meshVol, evals[k].hullVol), hd: evals[k].hd, cost: evals[k].cost, seam: evals[k].seam }));
        stats.pairsHausdorff += hdPairs.length;
        todo.forEach(([i, j], k) => cache.set(key(parts[i], parts[j]), evals[k]));
        // ---- greedy min-cost matching ----
        const ranked = candidates
            .map(([i, j]) => ({ i, j, ev: cache.get(key(parts[i], parts[j])) }))
            .filter((c) => c.ev.cost < opts.threshold && c.ev.hull)
            .sort((x, y) => x.ev.cost - y.ev.cost);
        const used = new Uint8Array(parts.length);
        const next = [];
        let merges = 0;
        for (const c of ranked) {
            if (used[c.i] || used[c.j])
                continue;
            used[c.i] = used[c.j] = 1;
            const A = parts[c.i], B = parts[c.j];
            const seams = [...(A.seams ?? []), ...(B.seams ?? [])];
            if (c.ev.seam)
                seams.push(c.ev.seam);
            next.push({
                mesh: mergeMeshes(A.mesh, B.mesh),
                hull: c.ev.hull,
                meshVol: A.meshVol + B.meshVol,
                hullVol: c.ev.hullVol,
                hausdorff: c.ev.hd,
                stuck: false,
                seams,
                merged: (A.merged ?? 1) + (B.merged ?? 1),
            });
            merges++;
        }
        if (opts.verbose) {
            console.log(`[merge round ${round}] parts=${parts.length} candidates=${candidates.length} hulled=${todo.length} hausdorff=${hdPairs.length} merged=${merges}`);
        }
        opts.onRound?.({ round, nParts: parts.length - merges, candidates: candidates.length, merged: merges });
        if (merges === 0)
            break;
        for (let i = 0; i < parts.length; i++)
            if (!used[i])
                next.push(parts[i]);
        parts = next;
        stats.merges += merges;
    }
    stats.totalMs = performance.now() - t0;
    return { parts, stats };
}
