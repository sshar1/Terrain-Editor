/** Plan *
 * Store 32 * 32 * 32 voxel grid initialized as a flat plane.
 * CPU takes 2D mouse position and computes the ray (NDC → model space).
 * Compute shader (mouseHit.wgsl) casts the ray against the grid to find the
 *   cursor position when Q is held.
 * Compute shader (update.wgsl) sculpts the grid using the cursor position.
 * Compute shader (marchingCubes.wgsl) extracts a triangle mesh from the grid.
 * Render shader (render.wgsl) draws the mesh with lighting.
 */

import * as Utils from './math_util.js';
import { TerrainRenderer } from './TerrainRenderer.js';
import { GridUpdater } from './GridUpdater.js';
import { VertexStreamer } from './VertexStreamer.js';

const VOXEL_RESOLUTION = 32;
const NUM_POINTS = VOXEL_RESOLUTION * VOXEL_RESOLUTION * VOXEL_RESOLUTION;

/**
 * Creates the initial scalar-field Float32Array for the GPU grid buffer.
 * Sets density = 1.0 for every voxel in the y = 1 slice, producing a
 * flat plane near the bottom of the [-1, 1]^3 volume.
 */
function createInitialGrid(resolution = VOXEL_RESOLUTION) {
    const data = new Float32Array(resolution * resolution * resolution);
    for (let z = 0; z < resolution; z++) {
        for (let x = 0; x < resolution; x++) {
            // flat index for (x, y=1, z)
            data[x + 1 * resolution + z * resolution * resolution] = 1.0;
        }
    }
    return data;
}

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

        // (Grid data is GPU-resident; initial upload happens in init())

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

    async createStorageBuffers() {
        const gridBufferSize = NUM_POINTS * 4;
        this.gridBuffer = this.device.createBuffer({
            size: gridBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        const cursorBufferSize = 16;
        this.cursorBuffer = this.device.createBuffer({
            size: cursorBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
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

        await this.createStorageBuffers();

        // Upload the initial flat-plane grid to the GPU
        this.device.queue.writeBuffer(this.gridBuffer, 0, createInitialGrid());

        // Initialize Renderers
        this.terrainRenderer = await TerrainRenderer.create(this.device, this.colorFormat, this.depthFormat);

        // Initialize computers
        this.gridUpdater = await GridUpdater.create(this.device, this.gridBuffer, this.cursorBuffer, VOXEL_RESOLUTION);
        this.vertexStreamer = await VertexStreamer.create(this.device, this.gridBuffer, this.gridUpdater.updateUniformBuffer, VOXEL_RESOLUTION);

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

    // Returns { origin, dir } both in model space (the same space as the grid).
    // Correct unprojection: NDC → view-space → world-space → model-space.
    getMouseRay(ndcX, ndcY) {
        // 1. View-space ray direction.
        //    For perspective proj P, a pixel at NDC (nx,ny) corresponds to
        //    view-space direction (nx/P[0], ny/P[5], -1) before normalisation.
        const P  = this.proj;
        const vx = ndcX / P[0];  // P[0] = f / aspect
        const vy = ndcY / P[5];  // P[5] = f

        // 2. World-space direction: inv(V_rot) * viewDir = V_rot^T * viewDir.
        //    In column-major V, row i of V_rot^T = column i of V_rot.
        //    out[i] = sum_j  V[i*4 + j] * v[j]
        const V  = this.view;
        const wx = V[0]*vx + V[1]*vy + V[2]*(-1.0);
        const wy = V[4]*vx + V[5]*vy + V[6]*(-1.0);
        const wz = V[8]*vx + V[9]*vy + V[10]*(-1.0);

        // Camera world position: -R^T * t  where t = V[12..14]
        const cpx = -(V[0]*V[12] + V[1]*V[13] + V[2]*V[14]);
        const cpy = -(V[4]*V[12] + V[5]*V[13] + V[6]*V[14]);
        const cpz = -(V[8]*V[12] + V[9]*V[13] + V[10]*V[14]);

        // 3. Model-space: inv(modelRotation) * v = modelRotation^T * v.
        //    Same formula as step 2 but with this.modelRotation.
        const MR = this.modelRotation;
        const ox = MR[0]*cpx + MR[1]*cpy + MR[2]*cpz;
        const oy = MR[4]*cpx + MR[5]*cpy + MR[6]*cpz;
        const oz = MR[8]*cpx + MR[9]*cpy + MR[10]*cpz;

        const dx = MR[0]*wx + MR[1]*wy + MR[2]*wz;
        const dy = MR[4]*wx + MR[5]*wy + MR[6]*wz;
        const dz = MR[8]*wx + MR[9]*wy + MR[10]*wz;

        const len = Math.hypot(dx, dy, dz);
        if (len < 1e-6) return { origin: [ox, oy, oz], dir: [0, 0, -1] };

        return {
            origin: [ox, oy, oz],
            dir:    [dx/len, dy/len, dz/len],
        };
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

        const ray = this.getMouseRay(input.ndcX, input.ndcY);

        // Update MV and MVP matrices
        Utils.mat4Multiply(this.mv, this.view, this.modelRotation);
        Utils.mat4Multiply(this.mvp, this.proj, this.mv);

        // Update normal matrix
        Utils.mat4InverseTranspose(this.norm, this.mv);

        // Update renderer uniform buffers
        const sculptEnabled = (window.inputState && window.inputState.sculptMode) || false;
        this.terrainRenderer.updateUniforms(this.mvp, this.lightDirWorld);
        this.gridUpdater.updateUniforms(this.mvp, ray.origin, ray.dir, VOXEL_RESOLUTION, sculptEnabled);
    }

    render() {
        this.resizeCanvas();

        // Reset the indirect draw counter on the GPU before execution
        this.device.queue.writeBuffer(this.vertexStreamer.indirectBuffer, 0, new Uint32Array([0, 1, 0, 0]));

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

        const computePassEncoder = commandEncoder.beginComputePass();
        this.gridUpdater.compute(computePassEncoder);
        this.vertexStreamer.compute(computePassEncoder);
        computePassEncoder.end();

        const renderPassEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
        this.terrainRenderer.draw(
            renderPassEncoder,
            this.vertexStreamer.vertexBuffer,
            this.vertexStreamer.indirectBuffer
        );
        renderPassEncoder.end();

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