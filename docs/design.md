# Design notes

This repository contains a TypeScript + WGSL port of the lookahead decomposition that
runs in browsers and in Deno. It was written as a hybrid: the tree search runs
on the GPU, the persistent decomposition is maintained on the CPU, and the
pieces that are awkward in WGSL (exact arithmetic, a device heap, warp
intrinsics) were redesigned rather than translated. See the README for
usage; this note records the design decisions and the validation that backs
them.

## Why not a straight translation

The CUDA code depends on three things WebGPU does not offer: 64/128-bit
integer arithmetic (the Bullet hull), a lock-based device heap with
device-scope fences (`allocator.cuh`), and a fixed 32-lane warp model with
`__syncwarp` (every warp-level helper). The port replaces each:

| CUDA | Port |
|------|------|
| exact-arithmetic D&C hull (`hull_dandc.cuh`, `kdop_hull.cuh`) | incremental quickhull per workgroup: 80 k-DOP extreme seeds, per-point owner face + farthest-point insertion (conflict lists), horizon from the edge-connected visible component. Orientation predicates are exact: coordinates are quantized to a 2^21 grid per part, the f32 determinant is trusted only beyond a rigorous rounding bound and a 96-bit integer determinant (three u32 limbs) decides otherwise. Validated against qhull on 1126 oblique-cut children of the bunny and teapot: worst relative volume error 2.3e-5 (quantization). Last-level branch-and-bound pruning as in `la_hull`. |
| arena heap, refcounted parts | bump allocation per tree level computed by a single-workgroup prefix scan over job counts; everything is dropped at the end of the iteration |
| capped plane cut per candidate (`plane_cut.cuh`, 1.5k lines) | uncapped cut; the cap only matters for the volume, which is recovered exactly from the open boundary loops (divergence-theorem anchor placed on the plane, on the line of two planes, or at the point of three) |
| per-candidate warp kernels, `__shfl`/`__syncwarp` | 256-thread workgroups, workgroup memory + `workgroupBarrier`, `workgroupUniformLoad` for every data-dependent loop bound (Tint's uniformity analysis is enforced by `tests/validate_tint.mjs`) |
| 10 blocking readbacks per iteration | two readbacks per iteration: the tree, its `evaluate` kernel and the refinement rounds are one submission; the hulls and Hausdorff distances of the new halves are another |
| CPU-side capping of winners | `cpu/cut.ts`: Sutherland–Hodgman clip, earcut caps with hole nesting by parity, centroid-fan fallback for degenerate rings |
| none | plane refinement after the tree: two batched scans along the chosen normal, each candidate scored with the full lookahead (CoACD's `clip_by_path` objective) |
| merge-hulls pass (`postprocess_merge.cu`, six kernels, one matching round) | `merge.ts`: rv and bbox rejections on the CPU, merged hulls of the two hulls' vertex sets and the seam-filtered Hausdorff as GPU batches (the existing hull and Hausdorff kernels, point-set seeds via part flag 2), greedy min-cost matching on the CPU, repeated with a per-pair cost cache until no pair is under the threshold |

The hull of a persistent part and the Hausdorff(part, hull) distance also run
on the GPU (same hull kernel with a face export; brute-force sampled
Hausdorff, one workgroup per 256 samples, same sampling rule as
`hausdorff.cuh`). A WASM build of Kuukka's quickhull is kept as a CPU
fallback; it misreports some voxel-mesh hulls (verified against qhull), so it
is not the default.

## Speed gaps found and closed

Measured on the bunny (RTX 4090, Deno) after the merge pass landed, the port
took 6.4 s against CUDA's 1.15 s. Profiling found these gaps, in order of
size:

1. **Hull kernel inserting the 80 k-DOP seeds in axis order.** On voxel
   meshes the extremes are often collinear, so the first faces were slivers
   with garbage normals and the surface broke; the kernel then inserted
   hundreds of points per child (iteration 0 spent 121 ms on 188 hulls, 40×
   the microbenchmark). Seeds are now inserted farthest-first like every
   other point.
2. **Float predicates.** Even with farthest-first order, 4% of the oblique
   (concave-edge) children got wrong hulls, some by 50% in volume, and no
   epsilon fixed it (one child breaks at 1e-4 and 1e-3, another passes only
   at 1e-4). The exact predicate layer removed every mismatch against qhull.
3. **Readback latency.** In Deno every `mapAsync` costs about 11 ms
   regardless of size; the pipeline did five per iteration (Hausdorff, tree,
   two refinement rounds, hulls), about 1.7 s on the bunny. The `evaluate`
   kernel reproduces `la_evaluate` on the GPU (path cost of every leaf,
   greedy fallback, refinement acceptance, next probe planes) so the tree and
   both refinement rounds are one submission; the hull kernel also writes a
   hull descriptor so the Hausdorff pass runs in the same command buffer.
4. **No last-level pruning.** `la_hull` skips the exact hull when the
   rough-hull lower bound cannot beat the best upper bound of the seed; the
   port now does the same (level-1 hull time 676 → 321 ms).
5. **Transfer overheads.** Hull exports reserved 96 KB per hull (200 MB
   readbacks in the merge pass), seeds were uploaded with one `writeBuffer`
   each, merge batches were small. Exports are sized by the batch's point
   count, uploads are packed, batches hold up to 1024 pairs.
6. **CPU edge structures.** `isClosed` and the concave-edge sampler used
   `Map`s keyed by 53-bit integers; a CSR adjacency replaced them (concave
   edges 180 → 65 ms, cut validation 460 → 290 ms including the cut itself).
7. **Seed search.** 40 axis passes with 80 workgroup reductions became 20
   passes with four batched reductions each (the kernel must stay under 16 KB
   of workgroup memory for Tint's default limit).

Result: bunny 6.4 → 2.0 s, STL 1.9 → 0.69 s, teapot 1.3 → 0.64 s, same part
counts. Of the remaining bunny time about 0.5 s is Deno readback latency
(44 round trips), 0.29 s the CPU cuts and 0.2 s the merge pass.

## Hull merging

The pair cost is CuACD's: `max(rv, Hausdorff)` of the merged hull against
the union of the two part meshes, with triangles on the seam ignored. The
seam test differs from both references on purpose. CoACD takes the first
face of A's hull that separates B, which is any face whose normal points
toward B, so it often filters nothing; CuACD's port requires every vertex
of B to lie *on* the plane, which only degenerate parts satisfy. The port
takes the separating face with the most coplanar mesh area, and a merged
part remembers its seams so later rounds ignore them too. Validation: on
the 285 parts of a CUDA bunny decomposition (dumped and re-evaluated with
`merge.ts`, one round), the pass produces exactly the 219 parts CUDA's
own merge pass produces; the remaining rounds take it to 210. The lower
merge ratio on the web decomposition itself (295 → 246) comes from the
refined part shapes, not from the pass: of ~1550 pairs that pass the rv
test on the bunny, ~1450 are rejected by the Hausdorff term with values
between 0.05 and 0.15, identical on the GPU and in the CPU reference.

## Pitfalls found on the way

- Vertices exactly in the cutting plane must get three-state treatment
  (`vsign`); the first "d ≥ 0 goes positive" rule put coplanar faces on one
  side and inflated the hull of the wrong half, so axis-aligned fixtures never
  reached cost 0.
- A float32 incremental hull that cones *every* visible face at once corrupts
  itself when float error makes the visible set disconnected; restricting to
  the component around the most visible face and inserting points
  farthest-first fixed it. Larger epsilons make it worse, not better.
- Face-slot compaction must never run between the visibility pass and the
  horizon pass (slot ids go stale).
- Earcut silently drops pieces of degenerate rings; the count `n + 2h − 2`
  detects it and a centroid fan twins every edge (zero-area loops included).
- Cap triangles must be oriented once per ring (from the largest triangle),
  not one by one: zero-area triangles flip randomly otherwise.
- Refining the plane with a two-halves *sum* objective raised the bunny from
  288 to 430+ parts; CoACD's path-averaged worst-part objective (what the
  tree already computes) brought it back to 280.
- The hull kernel skips a part without triangles unless flag 2 (no stored
  topology) is set; `hullMeshes` on raw point sets silently fell back to the
  WASM hull until `uploadSeeds` set it.
- The candidate list of the hull slab must hold at least the 80 seeds: with
  `nv` slots, parts under 80 points had their first face records overwritten
  and every small hull came back empty.
- Tint only accepts a branch that reaches a barrier or a `return` when its
  condition comes from `workgroupUniformLoad`; a value read back from a
  shared-memory reduction is not enough (`wgpu` accepts it).
- Timing `hullMeshes` by wall clock measures the readback: use the timestamp
  queries (`tree.profile`, `lastHullGpuMs`).
- The hull kernel's face export must be addressed by the global part index,
  not the chunk-local workgroup index: with several chunks per batch (the
  merge pass sends hundreds of point sets) later chunks overwrote earlier
  exports, merged parts got other hulls' faces, and the merge volumes taken
  from those faces skewed the matching.
- The exact grid hull is not a presentable mesh: points that are collinear or
  coplanar in float coordinates become hull vertices on the grid, so the raw
  export has needle faces and about a third more vertices than the true hull.
  `cleanHullFaces` rebuilds the output hull from the raw hull's vertex set in
  double precision with an epsilon above the float32 resolution of the input
  frame, drops vertices that sit within that height of a face (the CPU
  quickhull keeps early vertices that end up near-coplanar), and accepts the
  result only when it is closed, keeps the volume and contains the raw
  vertices (within 1e-4 of the extent; `tests/hull_output.test.ts`). The
  decomposition itself keeps using the kernel's exact volumes.

## Results

RTX 4090, Deno 2.9 (wgpu), `width 30, width2 3, nConcaveEdges 16,
threshold 0.05`, no preprocessing (voxel meshes made with trimesh at R=64):

| Mesh | tris | web parts (merged) / time | CUDA parts (merged) / time |
|------|------|---------------------------|-----------------------------|
| `tests/data/49160.stl` | 552 | 145 / 0.69 s | 144–148 / 0.14 s |
| teapot vox64 | 16.5k | 16 / 0.64 s | 24 / 0.29 s |
| bunny vox64 | 56k | 250 / 2.0 s | 223–225 / 1.15 s |

Bunny time breakdown: tree + refinement 1.0 s (22 submissions; hull kernels
about 0.5 s GPU), hulls + Hausdorff of new parts 0.36 s, CPU cuts 0.29 s,
merge 0.2 s, concave edges 0.06 s. Each submission still pays Deno's
readback latency (about 11 ms); Chrome's is far lower.

## Not ported / known gaps

- PaMO voxel preprocessing, connected-components splitting, `quick_depth`.
  Without component ids the merge pass may join parts of different
  connected components of the input (CuACD forbids that when
  `decompose_components` ran).
- Volumes of candidates bounded by three or more planes that do not share a
  point are approximate (depth ≥ 3 only; default depth 2 is exact).
- A part whose winning cut cannot be capped as a closed mesh is marked
  `stuck` and left alone (none occur on the three benchmark meshes now;
  `tests/cut_regress.test.ts` replays captured cases).
- The hull kernel costs 1–2 µs per small hull and up to 60 µs for a 27k-point
  part (throughput, 2048 in flight); the initial owner assignment is
  O(points × seed faces). `tests/bench_hull.ts` measures it per phase.

## Interactive demo

`demo/` (see the README) runs the decomposition in a dedicated
worker that owns the WebGPU device; the `Decomposer` reports granular
progress (`ProgressInfo.phase`, a monotone `fraction`) and honours an
`AbortSignal` at iteration and merge-round boundaries. Meshes are edited in
the browser with manifold-3d (booleans with primitives, twist, hull), which
guarantees closed input for the decomposer.

## Toolchain notes

- Deno 2.9 ships WebGPU (wgpu) and runs the kernels headless; Chrome could
  not be used on this machine (no Xvfb). wgpu's WGSL front end is more
  permissive than Tint, so `node --experimental-strip-types
  tests/validate_tint.mjs` (Dawn's `webgpu` npm package) is the gate for
  browser validity.
- `wasm/build.sh` needs emsdk (`~/.local/emsdk`); the generated `wasm/qh.js`
  is committed so Deno/browsers need no toolchain.
- `csrc/heap.c` got a CUDA 13 compatibility branch (`cuCtxCreate` grew a
  params argument) so the reference build works with the installed toolkit.
