// Validate the WGSL with Dawn/Tint (what Chrome uses) via the `webgpu` Node package.
// Usage: DAWN_NODE_DIR=/path/with/node_modules node --experimental-strip-types tests/validate_tint.mjs
//   (npm install webgpu in that directory first; defaults to /tmp/dawn_node)
const dawnDir = process.env.DAWN_NODE_DIR ?? "/tmp/dawn_node";
const { create, globals } = await import(`${dawnDir}/node_modules/webgpu/index.js`);
Object.assign(globalThis, globals);
const { SHADER_SOURCE } = await import("../src/gpu/shaders.ts");
const gpu = create([]);
const adapter = await gpu.requestAdapter();
if (!adapter) { console.log("no adapter (Dawn)"); process.exit(2); }
const device = await adapter.requestDevice();
device.addEventListener?.("uncapturederror", (e) => console.log("uncaptured:", e.error?.message));
device.pushErrorScope("validation");
const mod = device.createShaderModule({ code: SHADER_SOURCE });
const info = await mod.getCompilationInfo();
let bad = 0;
for (const m of info.messages) {
  if (m.type === "error") bad++;
  console.log(`${m.type} ${m.lineNum}:${m.linePos} ${m.message}`);
}
const err = await device.popErrorScope();
if (err) { console.log("validation:", err.message); bad++; }
if (!bad) {
  const layoutEntries = [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true } }];
  for (let b = 1; b <= 7; b++) layoutEntries.push({ binding: b, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } });
  const layout = device.createPipelineLayout({ bindGroupLayouts: [device.createBindGroupLayout({ entries: layoutEntries })] });
  for (const ep of ["plan", "scan", "emit", "hull", "record"]) {
    try {
      await device.createComputePipelineAsync({ layout, compute: { module: mod, entryPoint: ep } });
      console.log("pipeline ok:", ep);
    } catch (e) { console.log("pipeline FAIL:", ep, e.message); bad++; }
  }
}
console.log(bad ? "TINT: FAIL" : "TINT: OK");
process.exit(bad ? 1 : 0);
