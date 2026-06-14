async function createShaderModule(device, path) {
    const response = await fetch(path);
    const shaderSource = await response.text();
    return device.createShaderModule({
        code: shaderSource
    });
}

export class GridUpdater {
    static async create(device, gridBuffer, cursorBuffer, resolution) {
        const cursorHitShader   = await createShaderModule(device, "./shaders/mouseHit.wgsl");
        const updateShader      = await createShaderModule(device, "./shaders/update.wgsl");
        return new GridUpdater(device, cursorHitShader, updateShader, gridBuffer, cursorBuffer, resolution);
    }

    constructor(device, cursorHitShader, updateShader, gridBuffer, cursorBuffer, resolution) {
        this.device = device;
        this.resolution = resolution;

        // Uniform buffer for mouse hit (MVP: 16 floats, RayOrigin: 3 floats + 1 pad, RayDir: 3 floats + 1 pad = 24 floats = 96 bytes)
        this.hitUniformBuffer = device.createBuffer({
            size: 96,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Uniform buffer for grid update (BrushSize: 1 float, BrushStrength: 1 float, Resolution: 1 float + 1 pad = 4 floats = 16 bytes)
        this.updateUniformBuffer = device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Create pipelines using layout: "auto"
        this.hitPipeline = device.createComputePipeline({
            label: "Cursor Hit Pipeline",
            layout: "auto",
            compute: {
                module: cursorHitShader,
                entryPoint: "main",
            }
        });

        this.updatePipeline = device.createComputePipeline({
            label: "Grid Update Pipeline",
            layout: "auto",
            compute: {
                module: updateShader,
                entryPoint: "main",
            }
        });

        // Create bind groups querying layout from pipelines
        this.hitBindGroup = device.createBindGroup({
            layout: this.hitPipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: gridBuffer,
                    },
                },
                {
                    binding: 1,
                    resource: {
                        buffer: cursorBuffer,
                    },
                },
                {
                    binding: 2,
                    resource: {
                        buffer: this.hitUniformBuffer,
                    },
                },
            ],
        });

        this.updateBindGroup = device.createBindGroup({
            layout: this.updatePipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: gridBuffer,
                    },
                },
                {
                    binding: 1,
                    resource: {
                        buffer: cursorBuffer,
                    },
                },
                {
                    binding: 2,
                    resource: {
                        buffer: this.updateUniformBuffer,
                    },
                },
            ],
        });

        // Set default brush size and strength
        this.brushSize = 0.5;
        this.brushStrength = 0.1;
        this.setBrush(this.brushSize, this.brushStrength);
    }

    setBrush(size, strength) {
        this.brushSize = size;
        this.brushStrength = strength;
        const data = new Float32Array([this.brushSize, this.brushStrength, this.resolution, 0.0]);
        this.device.queue.writeBuffer(this.updateUniformBuffer, 0, data);
    }

    updateUniforms(mvp, ray_origin, ray_dir) {
        // mat(16) + ray_origin (3) + _pad0(1) + ray_dir(3) + _pad1(1)
        const uniformData = new Float32Array(24);
        uniformData.set(mvp, 0);
        uniformData.set(ray_origin, 16);
        uniformData.set(ray_dir, 20);
        this.device.queue.writeBuffer(this.hitUniformBuffer, 0, uniformData);
    }

    compute(passEncoder) {
        // 1. Run the cursor raycast hit detection (workgroup size is 1)
        passEncoder.setPipeline(this.hitPipeline);
        passEncoder.setBindGroup(0, this.hitBindGroup);
        passEncoder.dispatchWorkgroups(1);

        // 2. Run the grid update
        const numGroups = this.resolution / 8;
        passEncoder.setPipeline(this.updatePipeline);
        passEncoder.setBindGroup(0, this.updateBindGroup);
        passEncoder.dispatchWorkgroups(numGroups, numGroups, numGroups);

        passEncoder.end();
    }
}