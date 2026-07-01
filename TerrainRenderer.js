async function createShaderModule(device, path) {
    const response = await fetch(path);
    const shaderSource = await response.text();
    return device.createShaderModule({
        code: shaderSource
    });
}

export class TerrainRenderer {
    static async create(device, colorFormat, depthFormat) {
        const shaderModule = await createShaderModule(device, "./shaders/render.wgsl");
        return new TerrainRenderer(device, shaderModule, colorFormat, depthFormat);
    }

    constructor(device, shaderModule, colorFormat, depthFormat) {
        this.device = device;

        // Create uniform buffer: MVP mat4x4f (64 bytes) + LightDir vec3f (12 bytes) + pad (4 bytes) = 80 bytes
        this.uniformBuffer = device.createBuffer({
            size: 80,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const vertexLayout = [{
            attributes: [
                { shaderLocation: 0, offset: 0, format: 'float32x3' },  // position
                { shaderLocation: 1, offset: 12, format: 'float32x3' }, // normal
                { shaderLocation: 2, offset: 24, format: 'float32x3' }  // color
            ],
            arrayStride: 36, // 9 floats * 4 bytes
            stepMode: 'vertex'
        }];

        this.pipeline = device.createRenderPipeline({
            label: "Terrain Render Pipeline",
            layout: "auto",
            vertex: {
                module: shaderModule,
                entryPoint: "vertex_main",
                buffers: vertexLayout,
            },
            fragment: {
                module: shaderModule,
                entryPoint: "fragment_main",
                targets: [{ format: colorFormat }],
            },
            primitive: {
                topology: "triangle-list",
                cullMode: 'none',
            },
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: 'less',
                format: depthFormat,
            },
        });

        this.bindGroup = device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [{
                binding: 0,
                resource: { buffer: this.uniformBuffer }
            }]
        });

        this.vertexBuffer = null;
        this.indexBuffer = null;
        this.triangleCount = 0;
    }

    updateMesh(meshData) {
        if (this.vertexBuffer) this.vertexBuffer.destroy();
        if (this.indexBuffer) this.indexBuffer.destroy();

        this.triangleCount = meshData.triangleCount;

        if (meshData.vertexData.byteLength === 0) {
            this.vertexBuffer = null;
            this.indexBuffer = null;
            return;
        }

        this.vertexBuffer = this.device.createBuffer({
            size: meshData.vertexData.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.indexBuffer = this.device.createBuffer({
            size: meshData.triangleIndices.byteLength,
            usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        });

        this.device.queue.writeBuffer(this.vertexBuffer, 0, meshData.vertexData);
        this.device.queue.writeBuffer(this.indexBuffer, 0, meshData.triangleIndices);
    }

    updateUniforms(mvp, lightDir) {
        const uniformData = new Float32Array(20); // mvp(16), lightDirWorld(3), padding(1)
        uniformData.set(mvp, 0);
        uniformData.set(lightDir, 16);
        this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);
    }

    draw(passEncoder, vertexBuffer, indirectBuffer) {
        if (vertexBuffer && indirectBuffer) {
            passEncoder.setPipeline(this.pipeline);
            passEncoder.setBindGroup(0, this.bindGroup);
            passEncoder.setVertexBuffer(0, vertexBuffer);
            passEncoder.drawIndirect(indirectBuffer, 0);
            return;
        }

        if (!this.vertexBuffer || !this.indexBuffer || this.triangleCount === 0) return;

        passEncoder.setPipeline(this.pipeline);
        passEncoder.setBindGroup(0, this.bindGroup);
        passEncoder.setVertexBuffer(0, this.vertexBuffer);
        passEncoder.setIndexBuffer(this.indexBuffer, 'uint16');
        passEncoder.drawIndexed(this.triangleCount);
    }
}