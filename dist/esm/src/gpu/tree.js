// tree.ts — GPU lookahead tree search for one batch of cutting parts.
//
// The host uploads the parts to cut (seeds), records every level of the tree
// in one command buffer (plan → scan → emit → hull → record per level) and
// reads back only the small item table. Candidate meshes never leave the GPU.
import { ITEM_WORDS, JOB_WORDS, PARAMS_WORDS, PART_WORDS, SHADER_SOURCE, VIS_MAX } from "./shaders.js";
import { readBuffer, readBuffers } from "./device.js";
/** Hausdorff pairs per call (result slots in the counter buffer) */
export const HD_MAX_PAIRS = 1024;
export const DEFAULT_TREE_CONFIG = {
    width: 60,
    width2: 5,
    depth: 2,
    nConcaveEdges: 32,
    minEdgeDist: 0.0125,
    vertexCap: 64 << 20, // 256 MB
    triCap: 64 << 20, // 256 MB (leaf levels store no triangles)
    edgeCap: 1 << 20, // 32 MB
    scratchWords: 48 << 20, // 192 MB
    hullBudgetBytes: 256 << 20,
    hullMaxVerts: 4096,
    pruneLastLevel: true,
};
function meshArea(m) {
    const V = m.verts, T = m.tris;
    let area = 0;
    for (let t = 0; t < T.length; t += 3) {
        const a = T[t] * 3, b = T[t + 1] * 3, cc = T[t + 2] * 3;
        const abx = V[b] - V[a], aby = V[b + 1] - V[a + 1], abz = V[b + 2] - V[a + 2];
        const acx = V[cc] - V[a], acy = V[cc + 1] - V[a + 1], acz = V[cc + 2] - V[a + 2];
        area += 0.5 * Math.hypot(aby * acz - abz * acy, abz * acx - abx * acz, abx * acy - aby * acx);
    }
    return area;
}
function grid(n) {
    if (n <= 65535)
        return [Math.max(n, 1), 1];
    return [65535, Math.ceil(n / 65535)];
}
export class GpuTree {
    device;
    cfg;
    pipelines;
    layout;
    pool;
    parts;
    items;
    jobs;
    planes;
    scratch;
    ctr;
    params;
    bindGroup;
    capacity = { seeds: 0, jobs: 0, items: 0, parts: 0, planes: 0, paramSlots: 0 };
    edgeStride;
    ready;
    /** when true and the device supports timestamp queries, run() reports per-kernel GPU time */
    profile = false;
    ctrInit;
    querySet = null;
    queryBuf = null;
    constructor(device, cfg = {}) {
        this.device = device;
        this.cfg = { ...DEFAULT_TREE_CONFIG, ...cfg };
        if (this.cfg.width % 3 !== 0 || this.cfg.width2 % 3 !== 0)
            throw new Error("width and width2 must be multiples of 3");
        if (this.cfg.depth < 1 || this.cfg.depth > 4)
            throw new Error("depth must be in 1..4");
        // at least 32 slots so refinement probes work even without concave edges
        this.edgeStride = Math.max(4 * this.cfg.nConcaveEdges, 32);
        this.ready = this.init();
    }
    async init() {
        const d = this.device;
        const module = d.createShaderModule({ code: SHADER_SOURCE });
        const info = await module.getCompilationInfo();
        const errs = info.messages.filter((m) => m.type === "error");
        if (errs.length)
            throw new Error("WGSL compile errors:\n" + errs.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join("\n"));
        const entries = [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true } },
        ];
        for (let b = 1; b <= 7; b++)
            entries.push({ binding: b, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } });
        this.layout = d.createBindGroupLayout({ entries });
        const pl = d.createPipelineLayout({ bindGroupLayouts: [this.layout] });
        const mk = (entryPoint) => d.createComputePipelineAsync({ layout: pl, compute: { module, entryPoint } });
        const [plan, scan, emit, hull, record, hausdorff, evaluate] = await Promise.all([mk("plan"), mk("scan"), mk("emit"), mk("hull"), mk("record"), mk("hausdorff"), mk("evaluate")]);
        this.pipelines = { plan, scan, emit, hull, record, hausdorff, evaluate };
        this.ctrInit = d.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        const c = this.cfg;
        const poolBytes = (c.vertexCap + c.triCap + c.edgeCap * 8) * 4;
        this.pool = d.createBuffer({ size: poolBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
        this.scratch = d.createBuffer({
            size: c.scratchWords * 4 + c.hullBudgetBytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        this.ctr = d.createBuffer({ size: 4 * (8 + HD_MAX_PAIRS), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
        if (d.features.has("timestamp-query")) {
            this.querySet = d.createQuerySet({ type: "timestamp", count: 256 });
            this.queryBuf = d.createBuffer({ size: 256 * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
        }
        this.ensureTables(16);
    }
    /** (Re)allocate the per-run tables for up to `nSeeds` seeds. */
    /** Grow only the part table (for hull / Hausdorff batches that do not expand a tree). */
    ensurePartSlots(nParts) {
        if (nParts <= this.capacity.parts)
            return;
        const d = this.device;
        this.parts?.destroy();
        this.parts = d.createBuffer({ size: nParts * PART_WORDS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
        this.capacity.parts = nParts;
        this.rebuildBindGroup();
    }
    ensureTables(nSeeds) {
        if (nSeeds <= this.capacity.seeds)
            return;
        const d = this.device;
        const c = this.cfg;
        const w0 = c.width + this.edgeStride;
        let nItemsIn = nSeeds, totalJobs = 0;
        for (let l = 0; l < c.depth; l++) {
            const w = l === 0 ? w0 : c.width2;
            totalJobs += nItemsIn * w;
            nItemsIn *= w;
        }
        const nItems = nSeeds + totalJobs;
        const nParts = Math.max(nSeeds + 2 * totalJobs, this.capacity.parts);
        const nPlanes = totalJobs + nSeeds * this.edgeStride + 16 + 4 * nSeeds; // job planes | edge planes | evaluate state
        for (const b of [this.parts, this.items, this.jobs, this.planes])
            b?.destroy();
        const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
        this.parts = d.createBuffer({ size: nParts * PART_WORDS * 4, usage });
        this.items = d.createBuffer({ size: nItems * ITEM_WORDS * 4, usage });
        this.jobs = d.createBuffer({ size: Math.max(totalJobs, 1) * JOB_WORDS * 4, usage });
        this.planes = d.createBuffer({ size: Math.max(nPlanes, 1) * 16, usage });
        this.capacity = { seeds: nSeeds, jobs: totalJobs, items: nItems, parts: nParts, planes: nPlanes, paramSlots: 0 };
        this.rebuildBindGroup();
    }
    rebuildBindGroup() {
        const d = this.device;
        if (!this.params)
            this.params = d.createBuffer({ size: 64 * 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.bindGroup = d.createBindGroup({
            layout: this.layout,
            entries: [
                { binding: 0, resource: { buffer: this.params, size: PARAMS_WORDS * 4 } },
                { binding: 1, resource: { buffer: this.pool } },
                { binding: 2, resource: { buffer: this.parts } },
                { binding: 3, resource: { buffer: this.items } },
                { binding: 4, resource: { buffer: this.jobs } },
                { binding: 5, resource: { buffer: this.planes } },
                { binding: 6, resource: { buffer: this.scratch } },
                { binding: 7, resource: { buffer: this.ctr } },
            ],
        });
    }
    uploadSeeds(seeds) {
        const c = this.cfg;
        const d = this.device;
        const nSeeds = seeds.length;
        let vTop = 0, tTop = 0;
        const partWords = new Uint32Array(nSeeds * PART_WORDS);
        const partF = new Float32Array(partWords.buffer);
        // one packed upload for all vertices and one for all triangles (per-seed
        // writeBuffer calls cost more than the kernel for large batches)
        let totV = 0, totT = 0;
        for (const m of seeds) {
            totV += m.verts.length;
            totT += m.tris.length;
        }
        if (totV > c.vertexCap || totT > c.triCap)
            throw new Error("seed meshes exceed pool capacity");
        const packedV = new Float32Array(totV);
        const packedT = new Uint32Array(totT);
        for (let k = 0; k < nSeeds; k++) {
            const m = seeds[k];
            const nv = m.verts.length / 3, nt = m.tris.length / 3;
            packedV.set(m.verts, vTop);
            packedT.set(m.tris, tTop);
            const o = k * PART_WORDS;
            partWords[o] = vTop;
            partWords[o + 1] = nv;
            partWords[o + 2] = tTop;
            partWords[o + 3] = nt;
            partWords[o + 4] = 0;
            partWords[o + 5] = 0;
            partF[o + 6] = 0;
            partF[o + 7] = 0;
            partWords[o + 8] = partWords[o + 9] = partWords[o + 10] = partWords[o + 11] = 0xffffffff;
            partWords[o + 12] = 0;
            partWords[o + 13] = nt === 0 ? 2 : 0; // flag 2: point set without topology (hull only)
            vTop += m.verts.length;
            tTop += m.tris.length;
        }
        d.queue.writeBuffer(this.pool, 0, packedV);
        if (totT > 0)
            d.queue.writeBuffer(this.pool, c.vertexCap * 4, packedT);
        d.queue.writeBuffer(this.parts, 0, partWords);
        d.queue.writeBuffer(this.ctr, 0, new Uint32Array([vTop, tTop, 0, 0, 0, 0, 0, 0]));
        const itemWords = new Uint32Array(nSeeds * ITEM_WORDS);
        for (let k = 0; k < nSeeds; k++) {
            const o = k * ITEM_WORDS;
            itemWords[o] = k; // partIds[0]
            itemWords[o + 8] = 1; // np
            itemWords[o + 9] = k; // srcPart
            itemWords[o + 10] = 0xffffffff; // initialCut
            itemWords[o + 11] = 0; // nLevels
            itemWords[o + 16] = 1; // flags
            itemWords[o + 17] = 0xffffffff; // parentItem
        }
        d.queue.writeBuffer(this.items, 0, itemWords);
        return { vTop, tTop, partWords };
    }
    paramsWords(o) {
        const c = this.cfg;
        const u = new Uint32Array(PARAMS_WORDS);
        const f = new Float32Array(u.buffer);
        u[0] = o.itemsInBase;
        u[1] = o.nItemsIn;
        u[2] = o.itemsOutBase;
        u[3] = o.jobBase;
        u[4] = o.width;
        u[5] = o.nEdgeCuts;
        u[6] = this.edgeStride;
        u[7] = o.totalWidth;
        f[8] = c.minEdgeDist;
        u[9] = o.level;
        u[10] = o.totalLevels;
        u[11] = o.partBase;
        u[12] = 0;
        u[13] = c.vertexCap;
        u[14] = c.vertexCap;
        u[15] = c.triCap;
        u[16] = c.vertexCap + c.triCap;
        u[17] = c.edgeCap;
        u[18] = c.scratchWords;
        u[19] = o.edgePlaneBase;
        u[20] = o.hullChunkBase;
        u[21] = o.hullSlabWords;
        f[22] = 0;
        u[23] = o.nChildParts;
        u[24] = o.nJobs;
        u[25] = c.scratchWords;
        u[26] = o.hullChunkSize;
        u[27] = o.exportBase;
        u[28] = o.exportStride;
        u[29] = o.lastLevel ? 1 : 0;
        u[30] = o.hullStage ?? 0;
        u[31] = o.nSeeds ?? 0;
        return u;
    }
    hullSlab(seeds) {
        const c = this.cfg;
        let maxSeedNv = 0;
        for (const m of seeds)
            maxSeedNv = Math.max(maxSeedNv, m.verts.length / 3);
        // face slots: a hull over m points has < 2m faces; dead slots are bounded by
        // the compaction rule (nf <= 2 alive + 512 + 3 VIS_MAX); 16 words per face
        const fmax = 2 * Math.min(c.hullMaxVerts, maxSeedNv) + 3 * VIS_MAX + 1024;
        const hullSlabWords = ((2 * maxSeedNv + Math.max(maxSeedNv, 80) + 7) & ~7) + 16 * fmax;
        const hullChunkSize = Math.max(1, Math.floor(c.hullBudgetBytes / (hullSlabWords * 4)));
        this.lastHullSlabWords = hullSlabWords;
        this.lastHullChunkSize = hullChunkSize;
        return { hullSlabWords, hullChunkSize };
    }
    /**
     * Bidirectional sampled Hausdorff distance for mesh pairs (part, hull) on
     * the GPU. Pairs are uploaded as seeds: A_k at index k, B_k at n + k.
     */
    async hausdorff(pairs) {
        await this.ready;
        const c = this.cfg;
        const n = pairs.length;
        if (n === 0)
            return new Float32Array(0);
        this.ensureTables(1);
        this.ensurePartSlots(2 * n);
        const d = this.device;
        if (n > HD_MAX_PAIRS)
            throw new Error(`hausdorff: at most ${HD_MAX_PAIRS} pairs per call`);
        const seeds = [...pairs.map((p) => p.a), ...pairs.map((p) => p.b)];
        const { partWords } = this.uploadSeeds(seeds);
        // sample capacity per pair from the sampling rule's upper bound
        const bound = (m) => {
            const V = m.verts, T = m.tris;
            let area = 0;
            for (let t = 0; t < T.length; t += 3) {
                const a = T[t] * 3, b = T[t + 1] * 3, cc = T[t + 2] * 3;
                const abx = V[b] - V[a], aby = V[b + 1] - V[a + 1], abz = V[b + 2] - V[a + 2];
                const acx = V[cc] - V[a], acy = V[cc + 1] - V[a + 1], acz = V[cc + 2] - V[a + 2];
                area += 0.5 * Math.hypot(aby * acz - abz * acy, abz * acx - abx * acz, abx * acy - aby * acx);
            }
            const res = Math.max(1000, Math.floor(2000 * area));
            return 2 * res + T.length / 3 + 16;
        };
        const caps = pairs.map((p) => 2 * Math.max(bound(p.a), bound(p.b)));
        let off = 0, maxCap = 0;
        for (let k = 0; k < n; k++) {
            if (off + 3 * caps[k] > c.scratchWords)
                throw new Error("hausdorff sample scratch exceeds capacity");
            partWords[k * PART_WORDS + 4] = off; // eOff: sample scratch offset
            partWords[k * PART_WORDS + 5] = caps[k]; // ne: sample capacity
            off += 3 * caps[k];
            maxCap = Math.max(maxCap, caps[k]);
        }
        d.queue.writeBuffer(this.parts, 0, partWords.subarray(0, n * PART_WORDS));
        const slices = Math.ceil(maxCap / 256);
        d.queue.writeBuffer(this.ctr, 32, new Uint32Array(n));
        const params = this.paramsWords({
            itemsInBase: 0, nItemsIn: 0, itemsOutBase: 0, jobBase: 0, width: 0, nEdgeCuts: 0, totalWidth: 0, level: 0, totalLevels: 1,
            partBase: 0, edgePlaneBase: this.capacity.jobs, hullChunkBase: 0, hullSlabWords: 0, nChildParts: 0, nJobs: n, hullChunkSize: slices,
            exportBase: 0, exportStride: 0,
        });
        const pd = new Uint32Array(64);
        pd.set(params);
        d.queue.writeBuffer(this.params, 0, pd);
        const enc = d.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(this.pipelines.hausdorff);
        pass.setBindGroup(0, this.bindGroup, [0]);
        const [gx, gy] = grid(n * slices);
        pass.dispatchWorkgroups(gx, gy);
        pass.end();
        d.queue.submit([enc.finish()]);
        const res = new Float32Array(await readBuffer(d, this.ctr, 32, n * 4));
        return res.slice(0, n);
    }
    /**
     * Convex hulls of whole meshes on the GPU (same kernel as the tree search).
     * Returns outward-wound hull triangles over each mesh's own vertex indices,
     * or null for a mesh whose hull failed (caller falls back to the CPU).
     */
    /** GPU time of the last profiled hullMeshes call (ms), when `profile` is on */
    lastHullGpuMs = 0;
    async hullMeshes(meshes, opts = {}) {
        await this.ready;
        const c = this.cfg;
        const n = meshes.length;
        if (n === 0)
            return [];
        this.ensureTables(1);
        this.ensurePartSlots(n);
        const d = this.device;
        const { tTop } = this.uploadSeeds(meshes);
        const { hullSlabWords, hullChunkSize } = this.hullSlab(meshes);
        // a hull over m points has at most 2m - 4 faces: size the export per batch
        let maxNv = 0;
        for (const m of meshes)
            maxNv = Math.max(maxNv, m.verts.length / 3);
        const exportStride = opts.noExport ? 0 : 3 * 2 * Math.min(c.hullMaxVerts, maxNv);
        if (tTop + n * exportStride > c.triCap)
            throw new Error("hull export exceeds triangle pool");
        const slots = [];
        for (let base = 0; base < n; base += hullChunkSize) {
            slots.push(this.paramsWords({
                itemsInBase: 0, nItemsIn: n, itemsOutBase: n, jobBase: 0, width: 0, nEdgeCuts: 0, totalWidth: 0, level: 0, totalLevels: 1,
                partBase: 0, edgePlaneBase: this.capacity.jobs, hullChunkBase: base, hullSlabWords, nChildParts: n, nJobs: 0,
                hullChunkSize: Math.min(hullChunkSize, n - base), exportBase: tTop, exportStride, hullStage: opts.hullStage,
            }));
        }
        if (slots.length * 256 > this.params.size) {
            this.params.destroy();
            this.params = d.createBuffer({ size: slots.length * 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
            this.rebuildBindGroup();
        }
        const paramData = new Uint32Array(slots.length * 64);
        slots.forEach((u, i) => paramData.set(u, i * 64));
        d.queue.writeBuffer(this.params, 0, paramData);
        const enc = d.createCommandEncoder();
        const prof = this.profile && this.querySet !== null;
        const pass = enc.beginComputePass(prof ? { timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : {});
        pass.setPipeline(this.pipelines.hull);
        slots.forEach((_, i) => {
            pass.setBindGroup(0, this.bindGroup, [i * 256]);
            const [hx, hy] = grid(hullChunkSize);
            pass.dispatchWorkgroups(hx, hy);
        });
        pass.end();
        if (prof)
            enc.resolveQuerySet(this.querySet, 0, 2, this.queryBuf, 0);
        d.queue.submit([enc.finish()]);
        const reads = [{ buffer: this.parts, offset: 0, size: n * PART_WORDS * 4 }];
        if (exportStride > 0)
            reads.push({ buffer: this.pool, offset: (c.vertexCap + tTop) * 4, size: n * exportStride * 4 });
        if (prof)
            reads.push({ buffer: this.queryBuf, offset: 0, size: 16 });
        const bufs = await readBuffers(d, reads);
        const partsBuf = bufs[0];
        const triBuf = exportStride > 0 ? bufs[1] : undefined;
        if (prof) {
            const q = new BigUint64Array(bufs[bufs.length - 1]);
            this.lastHullGpuMs = Number(q[1] - q[0]) / 1e6;
        }
        const pw = new Uint32Array(partsBuf);
        const pf = new Float32Array(partsBuf);
        const tw = new Uint32Array(triBuf ?? new ArrayBuffer(0));
        const out = [];
        for (let k = 0; k < n; k++) {
            const vol = pf[k * PART_WORDS + 7];
            const nf = pw[k * PART_WORDS + 14];
            if (vol < 0 || nf < 4) {
                out.push(null);
                continue;
            }
            out.push({ tris: tw.slice(k * exportStride, k * exportStride + 3 * nf), volume: vol });
        }
        return out;
    }
    /**
     * Run the tree for the given seeds. `edgePlanes[k]` holds up to
     * 4*nConcaveEdges planes (a,b,c,d) for seed k; zero planes are skipped.
     */
    async run(seeds, edgePlanes = [], opts = {}) {
        await this.ready;
        const c = this.cfg;
        const depth = opts.depth ?? c.depth;
        const width0 = opts.width ?? c.width;
        const nEdge0 = Math.min(opts.nEdgeCuts ?? this.edgeStride, this.edgeStride);
        if (width0 % 3 !== 0 || width0 > c.width)
            throw new Error("width override must be a multiple of 3 and <= configured width");
        if (depth < 1 || depth > c.depth)
            throw new Error("depth out of range for this tree");
        const nSeeds = seeds.length;
        if (nSeeds === 0) {
            return { seeds: [], overflow: false, hullFail: false, hullFailCount: 0, npOverflow: false, levels: [[]], stats: { jobsPerLevel: [], gpuMs: 0, hullSlabWords: 0, hullChunkSize: 0 } };
        }
        const refine = opts.refine && opts.refine.rounds > 0 && opts.refine.probes >= 3 ? opts.refine : null;
        const nProbes = refine ? Math.min(refine.probes, this.edgeStride) : 0;
        this.ensureTables(nSeeds);
        const d = this.device;
        const { hullSlabWords, hullChunkSize } = this.hullSlab(seeds);
        // ---- upload seeds into the pool ----
        const { vTop, tTop } = this.uploadSeeds(seeds);
        d.queue.writeBuffer(this.ctrInit, 0, new Uint32Array([vTop, tTop, 0, 0]));
        // ---- level layouts (tree, and the probe rounds that reuse the tables) ----
        const layout = (w0, ne0) => {
            const levels = [];
            let itemsInBase = 0, nItemsIn = nSeeds, itemsOutBase = nSeeds, jobBase = 0;
            for (let l = 0; l < depth; l++) {
                const width = l === 0 ? w0 : c.width2;
                const nEdgeCuts = l === 0 ? ne0 : 0;
                const totalWidth = width + nEdgeCuts;
                const nJobs = nItemsIn * totalWidth;
                levels.push({ itemsInBase, nItemsIn, itemsOutBase, jobBase, partBase: nSeeds + 2 * jobBase, nJobs, width, nEdgeCuts, totalWidth });
                itemsInBase = itemsOutBase;
                nItemsIn = nJobs;
                itemsOutBase += nJobs;
                jobBase += nJobs;
            }
            return levels;
        };
        const treeLevels = layout(width0, nEdge0);
        const probeLevels = refine ? layout(0, nProbes) : null;
        const totalItems = treeLevels[treeLevels.length - 1].itemsOutBase + treeLevels[treeLevels.length - 1].nJobs;
        this.lastPartBase = treeLevels[0].partBase;
        const edgePlaneBase = this.capacity.jobs; // planes after all job planes
        const stateBase = edgePlaneBase + this.capacity.seeds * this.edgeStride; // evaluate state after the edge planes
        // edge planes
        const ep = new Float32Array(nSeeds * this.edgeStride * 4);
        for (let k = 0; k < nSeeds; k++) {
            const src = edgePlanes[k];
            if (src)
                ep.set(src.subarray(0, Math.min(src.length, this.edgeStride * 4)), k * this.edgeStride * 4);
        }
        if (ep.length)
            d.queue.writeBuffer(this.planes, edgePlaneBase * 16, ep);
        // evaluate state: two level-layout headers, then 4 vec4 per seed
        const stateWords = (16 + 4 * nSeeds) * 4;
        const state = new Float32Array(stateWords);
        const su = new Uint32Array(state.buffer);
        const writeHeader = (at, levels) => {
            su[at] = depth;
            su[at + 1] = nSeeds;
            su[at + 2] = levels[0].totalWidth;
            su[at + 3] = nProbes;
            levels.forEach((L, l) => { su[at + 4 + 4 * l] = L.itemsOutBase; su[at + 5 + 4 * l] = L.nJobs; });
        };
        writeHeader(0, treeLevels);
        if (probeLevels)
            writeHeader(20, probeLevels);
        for (let k = 0; k < nSeeds; k++)
            state[(16 + 4 * k + 2) * 4] = refine ? (refine.spacing[k] ?? 0) : 0;
        d.queue.writeBuffer(this.planes, stateBase * 16, state);
        // ---- params slots ----
        const slots = [];
        const push = (u) => { slots.push(u); return (slots.length - 1) * 256; };
        const mkLevel = (L, level, hullChunkBase, chunk) => push(this.paramsWords({
            itemsInBase: L.itemsInBase, nItemsIn: L.nItemsIn, itemsOutBase: L.itemsOutBase, jobBase: L.jobBase,
            width: L.width, nEdgeCuts: L.nEdgeCuts, totalWidth: L.totalWidth, level, totalLevels: depth, partBase: L.partBase,
            edgePlaneBase, hullChunkBase, hullSlabWords, nChildParts: 2 * L.nJobs, nJobs: L.nJobs, hullChunkSize: chunk,
            exportBase: 0, exportStride: 0, lastLevel: level === depth - 1, nSeeds: c.pruneLastLevel ? nSeeds : 0,
        }));
        const mkEval = (L0, mode, headerOff) => push(this.paramsWords({
            itemsInBase: 0, nItemsIn: 0, itemsOutBase: 0, jobBase: L0.jobBase, width: L0.width, nEdgeCuts: L0.nEdgeCuts, totalWidth: L0.totalWidth,
            level: 0, totalLevels: depth, partBase: 0, edgePlaneBase, hullChunkBase: stateBase, hullSlabWords: mode, nChildParts: headerOff,
            nJobs: 0, hullChunkSize: nProbes, exportBase: 0, exportStride: 0, nSeeds,
        }));
        const mkStage = (levels, mode, headerOff, tag) => {
            const levelSlot = [];
            const hullSlots = [];
            levels.forEach((L, l) => {
                levelSlot.push(mkLevel(L, l, 0, 0));
                const hs = [];
                const nChild = 2 * L.nJobs;
                for (let base = 0; base < nChild; base += hullChunkSize)
                    hs.push(mkLevel(L, l, base, Math.min(hullChunkSize, nChild - base)));
                hullSlots.push(hs);
            });
            return { levels, levelSlot, hullSlots, evalSlot: mkEval(levels[0], mode, headerOff), tag };
        };
        const stages = [mkStage(treeLevels, 0, 0, "")];
        if (probeLevels)
            for (let r = 0; r < refine.rounds; r++)
                stages.push(mkStage(probeLevels, 1, 20 / 4, `r${r}`));
        if (slots.length * 256 > this.params.size) {
            this.params.destroy();
            this.params = d.createBuffer({ size: slots.length * 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
            this.rebuildBindGroup();
        }
        const paramData = new Uint32Array(slots.length * 64);
        slots.forEach((u, i) => paramData.set(u, i * 64));
        d.queue.writeBuffer(this.params, 0, paramData);
        // ---- encode: tree, evaluate, then each probe round (counters reset, levels, evaluate) ----
        const t0 = performance.now();
        const enc = d.createCommandEncoder();
        const prof = this.profile && this.querySet !== null;
        const marks = [];
        let qi = 0;
        const beginPass = (name) => {
            if (!prof || qi + 2 > 256)
                return enc.beginComputePass();
            marks.push(name);
            const p = enc.beginComputePass({ timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: qi, endOfPassWriteIndex: qi + 1 } });
            qi += 2;
            return p;
        };
        stages.forEach((st, si) => {
            if (si > 0)
                enc.copyBufferToBuffer(this.ctrInit, 0, this.ctr, 0, 16);
            const pass = prof ? null : enc.beginComputePass();
            const kernel = (name, body) => {
                if (prof) {
                    const p = beginPass(name);
                    body(p);
                    p.end();
                }
                else
                    body(pass);
            };
            st.levels.forEach((L, l) => {
                const [gx, gy] = grid(L.nJobs);
                const tag = `${st.tag}${l}`;
                kernel(`plan${tag}`, (p) => { p.setPipeline(this.pipelines.plan); p.setBindGroup(0, this.bindGroup, [st.levelSlot[l]]); p.dispatchWorkgroups(gx, gy); });
                kernel(`scan${tag}`, (p) => { p.setPipeline(this.pipelines.scan); p.setBindGroup(0, this.bindGroup, [st.levelSlot[l]]); p.dispatchWorkgroups(1); });
                kernel(`emit${tag}`, (p) => { p.setPipeline(this.pipelines.emit); p.setBindGroup(0, this.bindGroup, [st.levelSlot[l]]); p.dispatchWorkgroups(gx, gy); });
                kernel(`hull${tag}`, (p) => {
                    p.setPipeline(this.pipelines.hull);
                    for (const slot of st.hullSlots[l]) {
                        p.setBindGroup(0, this.bindGroup, [slot]);
                        const [hx, hy] = grid(hullChunkSize);
                        p.dispatchWorkgroups(hx, hy);
                    }
                });
                kernel(`record${tag}`, (p) => { p.setPipeline(this.pipelines.record); p.setBindGroup(0, this.bindGroup, [st.levelSlot[l]]); p.dispatchWorkgroups(Math.ceil(L.nJobs / 256)); });
            });
            kernel(`eval${st.tag}`, (p) => { p.setPipeline(this.pipelines.evaluate); p.setBindGroup(0, this.bindGroup, [st.evalSlot]); p.dispatchWorkgroups(nSeeds); });
            if (pass)
                pass.end();
        });
        if (prof)
            enc.resolveQuerySet(this.querySet, 0, qi, this.queryBuf, 0);
        d.queue.submit([enc.finish()]);
        // ---- readback ----
        const lastLevels = stages[stages.length - 1].levels;
        const lastItems = lastLevels[lastLevels.length - 1].itemsOutBase + lastLevels[lastLevels.length - 1].nJobs;
        const reads = [
            { buffer: this.items, offset: 0, size: Math.max(totalItems, lastItems) * ITEM_WORDS * 4 },
            { buffer: this.ctr, offset: 0, size: 32 },
            { buffer: this.planes, offset: stateBase * 16, size: stateWords * 4 },
        ];
        if (opts.readParts)
            reads.push({ buffer: this.parts, offset: treeLevels[0].partBase * PART_WORDS * 4, size: 2 * treeLevels[0].nJobs * PART_WORDS * 4 });
        if (prof)
            reads.push({ buffer: this.queryBuf, offset: 0, size: Math.max(qi, 2) * 8 });
        const bufs = await readBuffers(d, reads);
        const itemsBuf = bufs[0], ctrBuf = bufs[1], stateBuf = bufs[2];
        const partsBuf = opts.readParts ? bufs[3] : null;
        const gpuMs = performance.now() - t0;
        let kernelMs;
        if (prof) {
            const q = new BigUint64Array(bufs[bufs.length - 1]);
            kernelMs = {};
            marks.forEach((name, i) => { kernelMs[name] = Number(q[2 * i + 1] - q[2 * i]) / 1e6; });
        }
        const iw = new Uint32Array(itemsBuf);
        const ifl = new Float32Array(itemsBuf);
        const ctr = new Uint32Array(ctrBuf);
        const sf = new Float32Array(stateBuf);
        const sw = new Uint32Array(stateBuf);
        const view = (i) => {
            const o = i * ITEM_WORDS;
            return {
                partIds: iw.subarray(o, o + 8),
                np: iw[o + 8], srcPart: iw[o + 9], initialCut: iw[o + 10], nLevels: iw[o + 11],
                levelCosts: ifl.subarray(o + 12, o + 16),
                flags: iw[o + 16], parentItem: iw[o + 17], cutIdx: iw[o + 18],
            };
        };
        // item views of the last encoded run (the tree when there was no refinement)
        const levelViews = [];
        let base = 0;
        const counts = [nSeeds, ...lastLevels.map((L) => L.nJobs)];
        for (const n of counts) {
            const arr = [];
            for (let i = 0; i < n; i++)
                arr.push(view(base + i));
            levelViews.push(arr);
            base += n;
        }
        // best cut per seed from the evaluate kernel's state
        const seedsOut = [];
        for (let k = 0; k < nSeeds; k++) {
            const o = (16 + 4 * k) * 4;
            const valid = (sw[o + 7] & 1) !== 0;
            seedsOut.push({
                bestCut: valid ? sw[o + 6] : -1,
                bestCost: valid ? sf[o + 4] : Infinity,
                plane: valid ? [sf[o], sf[o + 1], sf[o + 2], sf[o + 3]] : null,
            });
        }
        return {
            seeds: seedsOut,
            overflow: ctr[4] !== 0,
            hullFail: ctr[5] !== 0,
            hullFailCount: ctr[5],
            npOverflow: ctr[6] !== 0,
            levels: levelViews,
            stats: { jobsPerLevel: treeLevels.map((L) => L.nJobs), gpuMs, hullSlabWords, hullChunkSize, kernelMs, pruned: ctr[7] },
            partVols: partsBuf ? { base: treeLevels[0].partBase, data: new Float32Array(partsBuf) } : undefined,
        };
    }
    /**
     * Convex hulls and Hausdorff(part, hull) distances of whole meshes in one
     * submission: the hull kernel exports the faces and a hull descriptor, the
     * Hausdorff kernel samples both in the same command buffer.
     */
    /** failure codes of the last hullsWithHausdorff call (0 = ok, 1 = visible set overflow, 2 = face slots exhausted, 3 = slab too small) */
    lastHullFails = [];
    async hullsWithHausdorff(meshes) {
        await this.ready;
        const c = this.cfg;
        const n = meshes.length;
        if (n === 0)
            return [];
        if (n > HD_MAX_PAIRS)
            throw new Error(`hullsWithHausdorff: at most ${HD_MAX_PAIRS} meshes per call`);
        this.ensureTables(1);
        this.ensurePartSlots(2 * n);
        const d = this.device;
        const { tTop, partWords } = this.uploadSeeds(meshes);
        const { hullSlabWords, hullChunkSize } = this.hullSlab(meshes);
        let maxNv = 0;
        for (const m of meshes)
            maxNv = Math.max(maxNv, m.verts.length / 3);
        const exportStride = 3 * 2 * Math.min(c.hullMaxVerts, maxNv);
        if (tTop + n * exportStride > c.triCap)
            throw new Error("hull export exceeds triangle pool");
        // sample scratch per pair: the hull's area is at most the closed part's,
        // its triangle count at most 2 nv
        let off = 0, maxCap = 0;
        for (let k = 0; k < n; k++) {
            const m = meshes[k];
            const res = Math.max(1000, Math.floor(2000 * meshArea(m)));
            const cap = 2 * (2 * res + Math.max(m.tris.length / 3, 2 * (m.verts.length / 3)) + 16);
            if (off + 3 * cap > c.scratchWords)
                throw new Error("hausdorff sample scratch exceeds capacity");
            partWords[k * PART_WORDS + 4] = off;
            partWords[k * PART_WORDS + 5] = cap;
            off += 3 * cap;
            maxCap = Math.max(maxCap, cap);
        }
        d.queue.writeBuffer(this.parts, 0, partWords.subarray(0, n * PART_WORDS));
        d.queue.writeBuffer(this.ctr, 32, new Uint32Array(n));
        const slices = Math.ceil(maxCap / 256);
        const slots = [];
        for (let base = 0; base < n; base += hullChunkSize) {
            slots.push(this.paramsWords({
                itemsInBase: 0, nItemsIn: n, itemsOutBase: n, jobBase: 0, width: 0, nEdgeCuts: 0, totalWidth: 0, level: 0, totalLevels: 1,
                partBase: 0, edgePlaneBase: this.capacity.jobs, hullChunkBase: base, hullSlabWords, nChildParts: n, nJobs: n,
                hullChunkSize: Math.min(hullChunkSize, n - base), exportBase: tTop, exportStride,
            }));
        }
        const hdSlot = slots.length;
        slots.push(this.paramsWords({
            itemsInBase: 0, nItemsIn: 0, itemsOutBase: 0, jobBase: 0, width: 0, nEdgeCuts: 0, totalWidth: 0, level: 0, totalLevels: 1,
            partBase: 0, edgePlaneBase: this.capacity.jobs, hullChunkBase: 0, hullSlabWords: 0, nChildParts: 0, nJobs: n, hullChunkSize: slices,
            exportBase: 0, exportStride: 0,
        }));
        if (slots.length * 256 > this.params.size) {
            this.params.destroy();
            this.params = d.createBuffer({ size: slots.length * 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
            this.rebuildBindGroup();
        }
        const paramData = new Uint32Array(slots.length * 64);
        slots.forEach((u, i) => paramData.set(u, i * 64));
        d.queue.writeBuffer(this.params, 0, paramData);
        const enc = d.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(this.pipelines.hull);
        for (let i = 0; i < hdSlot; i++) {
            pass.setBindGroup(0, this.bindGroup, [i * 256]);
            const [hx, hy] = grid(hullChunkSize);
            pass.dispatchWorkgroups(hx, hy);
        }
        pass.setPipeline(this.pipelines.hausdorff);
        pass.setBindGroup(0, this.bindGroup, [hdSlot * 256]);
        const [gx, gy] = grid(n * slices);
        pass.dispatchWorkgroups(gx, gy);
        pass.end();
        d.queue.submit([enc.finish()]);
        const [partsBuf, triBuf, hdBuf] = await readBuffers(d, [
            { buffer: this.parts, offset: 0, size: n * PART_WORDS * 4 },
            { buffer: this.pool, offset: (c.vertexCap + tTop) * 4, size: n * exportStride * 4 },
            { buffer: this.ctr, offset: 32, size: n * 4 },
        ]);
        const pw = new Uint32Array(partsBuf);
        const pf = new Float32Array(partsBuf);
        const tw = new Uint32Array(triBuf);
        const hd = new Float32Array(hdBuf);
        const out = [];
        this.lastHullFails = [];
        for (let k = 0; k < n; k++) {
            const vol = pf[k * PART_WORDS + 7];
            const nf = pw[k * PART_WORDS + 14];
            if (vol < 0 || nf < 4) {
                out.push(null);
                this.lastHullFails.push(vol < 0 ? (pw[k * PART_WORDS + 13] & 3) || 9 : 8);
                continue;
            }
            out.push({ tris: tw.slice(k * exportStride, k * exportStride + 3 * nf), volume: vol, hausdorff: hd[k] });
        }
        return out;
    }
    /** Planes per seed available for edge/probe cuts. */
    get planeSlots() {
        return this.edgeStride;
    }
    /** Debug: read a part's descriptor and geometry back (uncapped mesh). */
    async readPart(partIdx) {
        const d = this.device;
        const pw = new Uint32Array(await readBuffer(d, this.parts, partIdx * PART_WORDS * 4, PART_WORDS * 4));
        const pf = new Float32Array(pw.buffer);
        const c = this.cfg;
        const verts = new Float32Array(await readBuffer(d, this.pool, pw[0] * 4, pw[1] * 12));
        const tris = new Uint32Array(await readBuffer(d, this.pool, (c.vertexCap + pw[2]) * 4, pw[3] * 12));
        const edges = new Float32Array(await readBuffer(d, this.pool, (c.vertexCap + c.triCap + pw[4] * 8) * 4, pw[5] * 32));
        return {
            mesh: { verts, tris }, edges, meshVol: pf[6], hullVol: pf[7], nPlanes: pw[12],
            planeIds: [pw[8], pw[9], pw[10], pw[11]].slice(0, pw[12]),
        };
    }
    /** Debug: alive hull faces of child `childIdx` (index within level 0) from the last run's slab. */
    async readHullFaces(childIdx) {
        const c = this.cfg;
        const slabWords = this.lastHullSlabWords;
        const chunk = this.lastHullChunkSize;
        const slot = childIdx % chunk;
        const base = (c.scratchWords + slot * slabWords) * 4;
        const words = new Uint32Array(await readBuffer(this.device, this.scratch, base, slabWords * 4));
        const part = new Uint32Array(await readBuffer(this.device, this.parts, (this.lastPartBase + childIdx) * PART_WORDS * 4, PART_WORDS * 4));
        const nv = part[1];
        const fbase = (3 * nv + 7) & ~7;
        const fmax = Math.floor((slabWords - fbase) / 8);
        const out = [];
        let nSlots = 0;
        for (let f = 0; f < fmax; f++) {
            const k = fbase + 8 * f;
            if (words[k + 3] === 1) {
                out.push(words[k], words[k + 1], words[k + 2]);
                nSlots = f + 1;
            }
        }
        return { faces: new Uint32Array(out), nSlots };
    }
    lastHullSlabWords = 0;
    lastHullChunkSize = 0;
    lastPartBase = 0;
    async readPlane(planeIdx) {
        const f = new Float32Array(await readBuffer(this.device, this.planes, planeIdx * 16, 16));
        return [f[0], f[1], f[2], f[3]];
    }
    get visMax() {
        return VIS_MAX;
    }
    destroy() {
        for (const b of [this.pool, this.parts, this.items, this.jobs, this.planes, this.scratch, this.ctr, this.params])
            b?.destroy();
    }
}
