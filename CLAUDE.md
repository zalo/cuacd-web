# CLAUDE.md

Guidance for Claude Code when working in this repository.

cuacd-web is a TypeScript + WGSL port of CuACD's lookahead convex decomposition (see README.md for
the overview and `docs/design.md` for the design, the pitfalls and the performance history). It runs
in browsers and in Deno 2.x; `demo/` is the interactive app (three.js + manifold-3d, decomposition in
a Web Worker).

## Commands

```bash
deno task test                                   # 34 tests; needs a WebGPU adapter (Deno has one built in)
deno task bench model.obj [--no-merge]           # profiled decomposition benchmark (closed meshes only)
deno run --allow-all tests/bench_hull.ts         # hull kernel throughput per phase, GPU timestamps
node --experimental-strip-types tests/validate_tint.mjs   # Tint/Dawn validation of the WGSL (needs DAWN_NODE_DIR)
deno task bundle && deno task serve              # demo on http://localhost:8000/ (bundle is generated, git-ignored)
deno task smoke                                  # headless-Chromium boot check of the demo page (no GPU needed)
deno task wasm                                   # rebuild wasm/qh.js (needs emsdk)
```

`tests/gpu_hull_real.test.ts` and `tests/cut_regress.test.ts` are opt-in and read `/tmp/meshes/`
(voxel-remeshed bunny/teapot made with trimesh); they skip when the files are missing.

## Rules

- Every WGSL loop bound or branch that reaches a barrier or a `return` must come from
  `workgroupUniformLoad`; a value read back from a shared-memory reduction is not enough. Deno's wgpu
  accepts code that Tint rejects, so run the Tint check after every shader edit.
- Keep each kernel under 16 KB of workgroup memory (Tint's default limit).
- Never put backticks in a WGSL comment: the shaders are JS template strings.
- Time GPU work with timestamp queries (`tree.profile`), not wall clock: a Deno readback costs ~11 ms
  regardless of size, so design for few round trips.
- The tree ranks cuts by rv cost only and stops on `max(rv, hausdorff)`; that mismatch is CoACD's and
  is deliberate.
- Output hulls go through `cleanHullFaces`; hull *volumes* for the cost model come from the GPU kernel.
- Decomposition is not bit-reproducible run to run; tests assert on aggregates (part-count ranges,
  volume, costs under the threshold, closedness), never exact geometry.
- The demo imports three and manifold-3d from CDN import maps and `./cuacd.bundle.js`; its worker test
  rebuilds the bundle. Headless Chromium without a GPU can only boot the page (`deno task smoke`).
