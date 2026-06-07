/** Plan *
 * Store 50 * 50 * 50 voxel initialized as flat plane
 * CPU takes 2d mouse position and gets the ray
 * Compute shader takes ray as a uniform (as well as brush info) and updates grid
 * Compute shader takes updated voxel buffer and outputs a vertex buffer with vertices and normals. This is where we use marching cubes
 * Render shader draws triangles using the generated vertex buffer
 * Separate render shader draws mouse position
*/

import * as Utils from './math_util.js';

const VOXEL_RESOLUTION = 30;
const NUM_POINTS = VOXEL_RESOLUTION * VOXEL_RESOLUTION * VOXEL_RESOLUTION;

async function createShaderModule(device, path) {
    const response = await fetch(path);
    const shaderSource = await response.text();
    return device.createShaderModule({
        code: shaderSource
    });
}

class VoxelGrid {
    constructor(resolution = VOXEL_RESOLUTION) {
        this.resolution = resolution;
        this.numPoints = resolution * resolution * resolution;
        this.data = new Array(this.numPoints).fill(0);
        this.initDefaultPlane();
    }

    set(val, x, y, z) {
        if (x >= this.resolution || y >= this.resolution || z >= this.resolution) return;
        if (x < 0 || y < 0 || z < 0) return;
        this.data[z * this.resolution ** 2 + y * this.resolution + x] = val;
    }

    get(x, y, z) {
        if (x >= this.resolution || y >= this.resolution || z >= this.resolution) return -1;
        if (x < 0 || y < 0 || z < 0) return -1;
        return this.data[z * this.resolution ** 2 + y * this.resolution + x];
    }

    initDefaultPlane() {
        // Initialize a flat plane at y = 1
        for (let z = 0; z < this.resolution; z++) {
            for (let x = 0; x < this.resolution; x++) {
                this.set(1, x, 1, z);
            }
        }
    }

    generateMesh() {
        const vertexData = [];
        const triangleIndices = [];
        const lineIndices = [];

        const h = 2.0 / this.resolution;
        const topColor = [0.3, 0.65, 0.3];
        const sideColor = [0.5, 0.4, 0.3];

        for (let z = 0; z < this.resolution; z++) {
            for (let y = 0; y < this.resolution; y++) {
                for (let x = 0; x < this.resolution; x++) {
                    if (this.get(x, y, z) === 1) {
                        const xMin = -1.0 + x * h;
                        const xMax = -1.0 + (x + 1) * h;
                        const yMin = -1.0 + y * h;
                        const yMax = -1.0 + (y + 1) * h;
                        const zMin = -1.0 + z * h;
                        const zMax = -1.0 + (z + 1) * h;

                        const faces = [];

                        // Front face (normal [0, 0, 1])
                        if (this.get(x, y, z + 1) !== 1) {
                            faces.push({
                                verts: [
                                    [xMin, yMin, zMax],
                                    [xMax, yMin, zMax],
                                    [xMax, yMax, zMax],
                                    [xMin, yMax, zMax]
                                ],
                                n: [0, 0, 1],
                                color: sideColor
                            });
                        }

                        // Back face (normal [0, 0, -1])
                        if (this.get(x, y, z - 1) !== 1) {
                            faces.push({
                                verts: [
                                    [xMax, yMin, zMin],
                                    [xMin, yMin, zMin],
                                    [xMin, yMax, zMin],
                                    [xMax, yMax, zMin]
                                ],
                                n: [0, 0, -1],
                                color: sideColor
                            });
                        }

                        // Top face (normal [0, 1, 0])
                        if (this.get(x, y + 1, z) !== 1) {
                            faces.push({
                                verts: [
                                    [xMin, yMax, zMax],
                                    [xMax, yMax, zMax],
                                    [xMax, yMax, zMin],
                                    [xMin, yMax, zMin]
                                ],
                                n: [0, 1, 0],
                                color: topColor
                            });
                        }

                        // Bottom face (normal [0, -1, 0])
                        if (this.get(x, y - 1, z) !== 1) {
                            faces.push({
                                verts: [
                                    [xMin, yMin, zMin],
                                    [xMax, yMin, zMin],
                                    [xMax, yMin, zMax],
                                    [xMin, yMin, zMax]
                                ],
                                n: [0, -1, 0],
                                color: sideColor
                            });
                        }

                        // Right face (normal [1, 0, 0])
                        if (this.get(x + 1, y, z) !== 1) {
                            faces.push({
                                verts: [
                                    [xMax, yMin, zMax],
                                    [xMax, yMin, zMin],
                                    [xMax, yMax, zMin],
                                    [xMax, yMax, zMax]
                                ],
                                n: [1, 0, 0],
                                color: sideColor
                            });
                        }

                        // Left face (normal [-1, 0, 0])
                        if (this.get(x - 1, y, z) !== 1) {
                            faces.push({
                                verts: [
                                    [xMin, yMin, zMin],
                                    [xMin, yMin, zMax],
                                    [xMin, yMax, zMax],
                                    [xMin, yMax, zMin]
                                ],
                                n: [-1, 0, 0],
                                color: sideColor
                            });
                        }

                        faces.forEach(face => {
                            const base = vertexData.length / 9; // 9 elements per vertex
                            face.verts.forEach(v => {
                                vertexData.push(...v);          // pos (3 floats)
                                vertexData.push(...face.n);     // normal (3 floats)
                                vertexData.push(...face.color); // color (3 floats)
                            });

                            lineIndices.push(base, base + 1, base + 1, base + 2, base + 2, base + 3, base + 3, base);
                            triangleIndices.push(base, base + 1, base + 2, base, base + 2, base + 3);
                        });
                    }
                }
            }
        }

        return {
            vertexData: new Float32Array(vertexData),
            triangleIndices: new Uint16Array(triangleIndices),
            lineIndices: new Uint16Array(lineIndices),
            triangleCount: triangleIndices.length,
            lineCount: lineIndices.length
        };
    }
}

class TerrainRenderer {
    static async create(device, colorFormat, depthFormat) {
        const shaderModule = await createShaderModule(device, "./render.wgsl");
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
                cullMode: 'back',
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

    draw(passEncoder) {
        if (!this.vertexBuffer || !this.indexBuffer || this.triangleCount === 0) return;

        passEncoder.setPipeline(this.pipeline);
        passEncoder.setBindGroup(0, this.bindGroup);
        passEncoder.setVertexBuffer(0, this.vertexBuffer);
        passEncoder.setIndexBuffer(this.indexBuffer, 'uint16');
        passEncoder.drawIndexed(this.triangleCount);
    }
}

// ============================================================
//  5. BOX / GIZMO RENDERER PIPELINE
// ============================================================
class BoxRenderer {
    static async create(device, colorFormat, depthFormat) {
        const shaderModule = await createShaderModule(device, "./box.wgsl");
        return new BoxRenderer(device, shaderModule, colorFormat, depthFormat);
    }

    constructor(device, shaderModule, colorFormat, depthFormat) {
        this.device = device;

        // Uniforms: MVP mat4x4f (64 bytes) + NormalMatrix mat4x4f (64 bytes) + LightDir vec3f (12 bytes) + pad (4 bytes) = 144 bytes
        this.uniformBuffer = device.createBuffer({
            size: 144,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const vertexLayout = [{
            attributes: [
                { shaderLocation: 0, offset: 0, format: 'float32x3' },  // position
                { shaderLocation: 1, offset: 12, format: 'float32x3' }, // color
                { shaderLocation: 2, offset: 24, format: 'float32x3' }  // normal
            ],
            arrayStride: 36, // 9 floats * 4 bytes
            stepMode: 'vertex'
        }];

        this.pipeline = device.createRenderPipeline({
            label: "Box Render Pipeline",
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
                cullMode: 'back',
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

        // Initialize static Box Mesh
        const boxMesh = this.generateBoxMesh();
        this.vertexBuffer = device.createBuffer({
            size: boxMesh.vertexData.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.indexBuffer = device.createBuffer({
            size: boxMesh.triangleIndices.byteLength,
            usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        });

        device.queue.writeBuffer(this.vertexBuffer, 0, boxMesh.vertexData);
        device.queue.writeBuffer(this.indexBuffer, 0, boxMesh.triangleIndices);
        this.triangleCount = boxMesh.triangleCount;
    }

    generateBoxMesh() {
        // 6 faces, each with 4 unique vertices (for flat normals), 2 triangles
        // We will build a single bottom face for this demo, or we can uncomment others if desired.
        const faceData = [
            // bottom – gold
            { verts: [[-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]], n: [0, -1, 0], color: [1.00, 0.75, 0.20] }, 
        ];

        const vertexData = [];
        const triangleIndices = [];
        const lineIndices = [];

        faceData.forEach((face, i) => {
            const base = i * 4;
            face.verts.forEach(v => {
                vertexData.push(...v);          // pos (3)
                vertexData.push(...face.color); // color (3)
                vertexData.push(...face.n);     // normal (3)
            });

            lineIndices.push(base, base + 1, base + 1, base + 2, base + 2, base + 3, base + 3, base);
            triangleIndices.push(base, base + 1, base + 2, base, base + 2, base + 3);
        });

        return {
            vertexData: new Float32Array(vertexData),
            triangleIndices: new Uint16Array(triangleIndices),
            lineIndices: new Uint16Array(lineIndices),
            triangleCount: triangleIndices.length,
            lineCount: lineIndices.length
        };
    }

    updateUniforms(mvp, normalMatrix, lightDir) {
        const uniformData = new Float32Array(36); // mvp(16), normalMatrix(16), lightDirWorld(3), padding(1)
        uniformData.set(mvp, 0);
        uniformData.set(normalMatrix, 16);
        uniformData.set(lightDir, 32);
        this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);
    }

    draw(passEncoder) {
        passEncoder.setPipeline(this.pipeline);
        passEncoder.setBindGroup(0, this.bindGroup);
        passEncoder.setVertexBuffer(0, this.vertexBuffer);
        passEncoder.setIndexBuffer(this.indexBuffer, 'uint16');
        passEncoder.drawIndexed(this.triangleCount);
    }
}

// ============================================================
//  6. MAIN ENGINE COORDINATOR
// ============================================================
class Engine {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) {
            throw new Error(`Canvas with ID "${canvasId}" not found.`);
        }
        
        this.device = null;
        this.context = null;
        this.colorFormat = null;
        this.depthFormat = 'depth24plus';
        this.depthTexture = null;

        // Renderers
        this.terrainRenderer = null;
        this.boxRenderer = null;

        // Voxel Grid
        this.voxelGrid = new VoxelGrid();

        // Matrices
        this.proj = Utils.mat4Create();
        this.view = Utils.mat4Create();
        this.mv = Utils.mat4Create();
        this.mvp = Utils.mat4Create();
        this.norm = Utils.mat4Create();

        // Persistent model rotation matrix
        this.modelRotation = Utils.mat4Create();
        this.tempMatrix = Utils.mat4Create();

        // Initial rotation tilt
        Utils.mat4RotateX(this.modelRotation, this.modelRotation, 0.35);

        this.lightDirWorld = [0.0, 0.7, 1.0];
    }

    async init() {
        if (!navigator.gpu) {
            throw new Error("WebGPU not supported in this browser!");
        }

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            throw new Error("No appropriate GPUAdapter found!");
        }

        this.device = await adapter.requestDevice();
        this.context = this.canvas.getContext("webgpu");
        this.colorFormat = navigator.gpu.getPreferredCanvasFormat();

        // Configure Canvas Context
        this.resizeCanvas();
        this.context.configure({
            device: this.device,
            format: this.colorFormat,
            alphaMode: "premultiplied"
        });

        // Initialize Renderers
        this.terrainRenderer = await TerrainRenderer.create(this.device, this.colorFormat, this.depthFormat);
        this.boxRenderer = await BoxRenderer.create(this.device, this.colorFormat, this.depthFormat);

        // Setup Voxel Mesh
        this.updateVoxelMesh();

        // Start frame loop
        this.start();
    }

    resizeCanvas() {
        const dpr = window.devicePixelRatio || 1;
        const w = (this.canvas.clientWidth * dpr) | 0;
        const h = (this.canvas.clientHeight * dpr) | 0;

        if (this.canvas.width !== w || this.canvas.height !== h) {
            this.canvas.width = w;
            this.canvas.height = h;

            // Recreate depth texture on resize
            if (this.depthTexture) this.depthTexture.destroy();
            this.depthTexture = this.device.createTexture({
                size: [w, h],
                format: this.depthFormat,
                usage: GPUTextureUsage.RENDER_ATTACHMENT,
            });

            // Update projection matrix
            const aspect = w / h;
            Utils.mat4Perspective(this.proj, Math.PI / 6, aspect, 0.1, 100);
        }
    }

    updateVoxelMesh() {
        const meshData = this.voxelGrid.generateMesh();
        this.terrainRenderer.updateMesh(meshData);
    }

    getMouse3D(camPos, ndcX, ndcY) {
        const rayDir = [ndcX, ndcY, 0];
        rayDir[0] -= camPos[0];
        rayDir[1] -= camPos[1];
        rayDir[2] -= camPos[2];
        // Placeholder console log matched from original engine.js
        console.log("RayDir:", rayDir);
    }

    update() {
        const input = window.inputState || { 
            deltaX: 0, deltaY: 0, velocityX: 0, velocityY: 0, 
            zoom: 7, interacting: false, mouseX: 0, mouseY: 0, 
            ndcX: 0, ndcY: 0 
        };

        let dx = 0, dy = 0;
        if (input.interacting) {
            dx = input.deltaX;
            dy = input.deltaY;
            input.deltaX = 0;
            input.deltaY = 0;
        } else {
            dx = input.velocityX;
            dy = input.velocityY;
            input.velocityX *= 0.95;
            input.velocityY *= 0.95;
            if (Math.abs(input.velocityX) < 0.0001) input.velocityX = 0;
            if (Math.abs(input.velocityY) < 0.0001) input.velocityY = 0;
        }

        if (dx !== 0 || dy !== 0) {
            const inc = Utils.mat4Create();
            Utils.mat4RotateY(inc, inc, dx);
            Utils.mat4RotateX(inc, inc, dy);
            Utils.mat4Multiply(this.tempMatrix, inc, this.modelRotation);
            this.modelRotation.set(this.tempMatrix);
        }

        // Camera positioning with zoom
        const camPos = [0, 0, input.zoom];
        Utils.mat4LookAt(this.view, camPos, [0, 0, 0], [0, 1, 0]);

        this.getMouse3D(camPos, input.ndcX, input.ndcY);

        // Update MV and MVP matrices
        Utils.mat4Multiply(this.mv, this.view, this.modelRotation);
        Utils.mat4Multiply(this.mvp, this.proj, this.mv);

        // Update normal matrix
        Utils.mat4InverseTranspose(this.norm, this.mv);

        // Update renderer uniform buffers
        this.terrainRenderer.updateUniforms(this.mvp, this.lightDirWorld);
        this.boxRenderer.updateUniforms(this.mvp, this.norm, this.lightDirWorld);
    }

    render() {
        this.resizeCanvas();

        const commandEncoder = this.device.createCommandEncoder();
        const renderPassDescriptor = {
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                clearValue: { r: 0.03, g: 0.03, b: 0.05, a: 1.0 },
                loadOp: "clear",
                storeOp: "store",
            }],
            depthStencilAttachment: {
                view: this.depthTexture.createView(),
                depthClearValue: 1.0,
                depthLoadOp: 'clear',
                depthStoreOp: 'store',
            },
        };

        const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
        
        // Draw terrain
        this.terrainRenderer.draw(passEncoder);

        // Draw box (gizmo) if desired
        // this.boxRenderer.draw(passEncoder);

        passEncoder.end();
        this.device.queue.submit([commandEncoder.finish()]);
    }

    start() {
        const frame = () => {
            this.update();
            this.render();
            requestAnimationFrame(frame);
        };
        requestAnimationFrame(frame);
    }
}

// Initialize the Engine on Page Load
window.addEventListener("DOMContentLoaded", () => {
    const engine = new Engine("canvas");
    engine.init().catch(err => {
        console.error("Failed to initialize WebGPU Engine:", err);
    });
});