export interface GpuInfo {
    vendor: string;
    description: string;
    maxBufferSize: number;
    maxStorageBufferBindingSize: number;
    workgroupStorage: number;
}
export declare function getDevice(): Promise<{
    device: GPUDevice;
    info: GpuInfo;
}>;
export declare function readBuffer(device: GPUDevice, src: GPUBuffer, offset: number, bytes: number): Promise<ArrayBuffer>;
/** Read several buffer ranges with one staging buffer, one submit and one map. */
export declare function readBuffers(device: GPUDevice, reads: {
    buffer: GPUBuffer;
    offset: number;
    size: number;
}[]): Promise<ArrayBuffer[]>;
