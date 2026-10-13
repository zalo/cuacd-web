// device.ts — WebGPU adapter/device acquisition with the limits the tree
// search needs (large storage buffers). Works in browsers and in Deno.

export interface GpuInfo {
  vendor: string;
  description: string;
  maxBufferSize: number;
  maxStorageBufferBindingSize: number;
  workgroupStorage: number;
}

export async function getDevice(): Promise<{ device: GPUDevice; info: GpuInfo }> {
  const gpu = (globalThis as any).navigator?.gpu as GPU | undefined;
  if (!gpu) throw new Error("WebGPU is not available in this runtime");
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("No WebGPU adapter");
  const L = adapter.limits;
  const requiredLimits: Record<string, number> = {
    maxBufferSize: L.maxBufferSize,
    maxStorageBufferBindingSize: L.maxStorageBufferBindingSize,
    maxComputeWorkgroupStorageSize: Math.max(16384, L.maxComputeWorkgroupStorageSize),
    maxComputeInvocationsPerWorkgroup: Math.max(256, L.maxComputeInvocationsPerWorkgroup),
    maxStorageBuffersPerShaderStage: Math.max(8, L.maxStorageBuffersPerShaderStage),
  };
  const requiredFeatures: GPUFeatureName[] = [];
  if (adapter.features.has("timestamp-query")) requiredFeatures.push("timestamp-query");
  const device = await adapter.requestDevice({ requiredLimits, requiredFeatures });
  const info = {
    vendor: adapter.info?.vendor ?? "?",
    description: adapter.info?.description ?? "?",
    maxBufferSize: L.maxBufferSize,
    maxStorageBufferBindingSize: L.maxStorageBufferBindingSize,
    workgroupStorage: L.maxComputeWorkgroupStorageSize,
  };
  return { device, info };
}

export async function readBuffer(device: GPUDevice, src: GPUBuffer, offset: number, bytes: number): Promise<ArrayBuffer> {
  const size = Math.ceil(bytes / 4) * 4;
  const rb = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(src, offset, rb, 0, size);
  device.queue.submit([enc.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const out = rb.getMappedRange().slice(0);
  rb.unmap();
  rb.destroy();
  return out;
}

/** Read several buffer ranges with one staging buffer, one submit and one map. */
export async function readBuffers(device: GPUDevice, reads: { buffer: GPUBuffer; offset: number; size: number }[]): Promise<ArrayBuffer[]> {
  const sizes = reads.map((r) => Math.ceil(r.size / 4) * 4);
  const offsets: number[] = [];
  let total = 0;
  for (const s of sizes) { offsets.push(total); total += Math.ceil(s / 256) * 256; }
  const rb = device.createBuffer({ size: Math.max(total, 4), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = device.createCommandEncoder();
  reads.forEach((r, i) => { if (sizes[i] > 0) enc.copyBufferToBuffer(r.buffer, r.offset, rb, offsets[i], sizes[i]); });
  device.queue.submit([enc.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const all = rb.getMappedRange();
  const out = reads.map((_, i) => all.slice(offsets[i], offsets[i] + sizes[i]));
  rb.unmap();
  rb.destroy();
  return out;
}
