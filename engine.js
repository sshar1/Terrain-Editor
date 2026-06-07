import * as Utils from './math_util.js'

/** Plan *
 * Store 50 * 50 * 50 voxel initialized as flat plane
 * CPU calculates 3d mouse position by raycasting through voxel
 * Compute shader takes voxel and mouse position, updates voxel and outputs it to next shader
 * Actually, do this on cpu, not shader
 * Compute shader takes updated voxel buffer and outputs a vertex buffer with vertices and normals. This is where we use marching cubes
 * Render shader draws triangles using the generated vertex buffer
 * Separate render shader (?) draws mouse position
*/

const voxelResolution = 30;
const numPoints = voxelResolution * voxelResolution * voxelResolution;
const clearColor = { r: 0.03, g: 0.03, b: 0.05, a: 0.0 };

function setVoxelPoint(voxel, val, x, y, z) {
    if (x >= voxelResolution || y >= voxelResolution || z >= voxelResolution) return;
    if (x < 0|| y < 0 || z < 0) return;

    voxel[z*voxelResolution**2 + y*voxelResolution + x] = val
}

function getVoxelPoint(voxel, x, y, z) {
    if (x >= voxelResolution || y >= voxelResolution || z >= voxelResolution) return -1;
    if (x < 0|| y < 0 || z < 0) return -1;

    return voxel[z*voxelResolution**2 + y*voxelResolution + x]
}

// TODO voxel matrix should have continuous values; marching cubes will actual vertex positions
// for debugging, we will just use values of 1 or 0
function initVoxel() {
    const voxel = new Array(numPoints).fill(0);

    // From starting configuration:
    // +X -> Right
    // +Y -> Up
    // +Z -> Towards camera

    for (let z = 0; z < voxelResolution; z++) {
        for (let x = 0; x < voxelResolution; x++) {
            setVoxelPoint(voxel, 1, x, 1, z);
        }
    }
    return voxel;
}

function DEBUG_getVoxelVertices(voxel) {
    const vertexData = [];
    const triangle_indices = [];
    const line_indices = [];

    const h = 2.0 / voxelResolution;
    const topColor = [0.3, 0.65, 0.3];
    const sideColor = [0.5, 0.4, 0.3];

    for (let z = 0; z < voxelResolution; z++) {
        for (let y = 0; y < voxelResolution; y++) {
            for (let x = 0; x < voxelResolution; x++) {
                if (getVoxelPoint(voxel, x, y, z) === 1) {
                    const xMin = -1.0 + x * h;
                    const xMax = -1.0 + (x + 1) * h;
                    const yMin = -1.0 + y * h;
                    const yMax = -1.0 + (y + 1) * h;
                    const zMin = -1.0 + z * h;
                    const zMax = -1.0 + (z + 1) * h;

                    const faces = [];

                    // Front face (normal [0, 0, 1])
                    if (getVoxelPoint(voxel, x, y, z + 1) !== 1) {
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
                    if (getVoxelPoint(voxel, x, y, z - 1) !== 1) {
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
                    if (getVoxelPoint(voxel, x, y + 1, z) !== 1) {
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
                    if (getVoxelPoint(voxel, x, y - 1, z) !== 1) {
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
                    if (getVoxelPoint(voxel, x + 1, y, z) !== 1) {
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
                    if (getVoxelPoint(voxel, x - 1, y, z) !== 1) {
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

                        line_indices.push(base, base + 1, base + 1, base + 2, base + 2, base + 3, base + 3, base);
                        triangle_indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
                    });
                }
            }
        }
    }

    return {
        vertexData: new Float32Array(vertexData),
        triangle_indices: new Uint16Array(triangle_indices),
        line_indices: new Uint16Array(line_indices),
        triangle_count: triangle_indices.length,
        line_count: line_indices.length
    };
}

// Takes 2d mouse position, raycasts and returns the voxel position it hits
async function getMouse3D(cam_pos, mouseX, mouseY) {
    const rayDir = [mouseX, mouseY, 0];
    rayDir[0] -= cam_pos[0];
    rayDir[1] -= cam_pos[1];
    rayDir[2] -= cam_pos[2];
    
    console.log(rayDir);
}

function createBox() {
    // 6 faces, each with 4 unique vertices (for flat normals), 2 triangles
    const faceData = [
        // positions (4 verts)           normal           color
        // { verts: [[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], n: [0, 0, 1], color: [0.30, 0.60, 1.00] },  // front  – blue
        // { verts: [[1, -1, -1], [-1, -1, -1], [-1, 1, -1], [1, 1, -1]], n: [0, 0, -1], color: [0.20, 0.80, 0.50] },  // back   – green
        // { verts: [[-1, 1, 1], [1, 1, 1], [1, 1, -1], [-1, 1, -1]], n: [0, 1, 0], color: [1.00, 0.35, 0.40] },  // top    – red
        { verts: [[-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]], n: [0, -1, 0], color: [1.00, 0.75, 0.20] },  // bottom – gold
        // { verts: [[1, -1, 1], [1, -1, -1], [1, 1, -1], [1, 1, 1]], n: [1, 0, 0], color: [0.85, 0.30, 0.90] },  // right  – purple
        // { verts: [[-1, -1, -1], [-1, -1, 1], [-1, 1, 1], [-1, 1, -1]], n: [-1, 0, 0], color: [0.10, 0.85, 0.85] },  // left   – cyan
    ];

    const vertexData = [];
    const triangle_indices = [];
    const line_indices = [];

    faceData.forEach((face, i) => {
        const base = i * 4;
        face.verts.forEach(v => {
            vertexData.push(...v);          // pos (3)
            vertexData.push(...face.color); // color (3)
            vertexData.push(...face.n);     // normal (3)
        });

        line_indices.push(base, base + 1, base + 1, base + 2, base + 2, base + 3, base + 3, base);
        triangle_indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
    });

    return {
        vertexData: new Float32Array(vertexData),
        triangle_indices: new Uint16Array(triangle_indices),
        line_indices: new Uint16Array(line_indices),
        triangle_count: triangle_indices.length,
        line_count: line_indices.length
    };
}

async function createShaderModule(device, path) {
    const response = await fetch(path);
    const shaderSource = await response.text();
    return device.createShaderModule({
        code: shaderSource
    })
}

async function init() {
    if (!navigator.gpu) {
        throw new Error("WebGPU not supported!");
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        throw new Error("No appropriate GPUAdapter found!");
    }
    const device = await adapter.requestDevice();
    const canvas = document.getElementById("canvas");

    // Resize canvas to match display size (handles high-DPI)
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth * dpr | 0;
    const h = canvas.clientHeight * dpr | 0;
    if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
    }

    // Set up context
    const context = canvas.getContext("webgpu")
    context.configure({
        device: device,
        format: navigator.gpu.getPreferredCanvasFormat(),
        alphaMode: "premultiplied"
    })

    // Set up shader modules
    const terrainShaderModule = await createShaderModule(device, "./render.wgsl");

    // Uniforms: mat4x4 (64) vec3 (12) + pad (4) = 80
    const renderUniformBufferSize = 80;
    const uniformBuffer = device.createBuffer({
        size: renderUniformBufferSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const voxel = initVoxel();
    const voxelMesh = DEBUG_getVoxelVertices(voxel);

    const voxelVertexBuffer = device.createBuffer({
        size: voxelMesh.vertexData.byteLength,
        usage: GPUBufferUsage.VERTEX |  GPUBufferUsage.COPY_DST,
    });
    const voxelIndicesBuffer = device.createBuffer({
        size: voxelMesh.triangle_indices.byteLength,
        usage: GPUBufferUsage.INDEX |  GPUBufferUsage.COPY_DST,
    });

    device.queue.writeBuffer(voxelVertexBuffer, 0, voxelMesh.vertexData);
    device.queue.writeBuffer(voxelIndicesBuffer, 0, voxelMesh.triangle_indices);

    const voxelVertexBuffersLayout = [{
        attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x3' },  // position
            { shaderLocation: 1, offset: 12, format: 'float32x3' }, // normal
            { shaderLocation: 2, offset: 24, format: 'float32x3' }  // color
        ],
        arrayStride: 36, // 9 floats * 4 bytes
        stepMode: 'vertex'
    }];

    const voxelPipeline = device.createRenderPipeline({
        layout: "auto",
        vertex: {
            module: terrainShaderModule,
            entryPoint: "vertex_main",
            buffers: voxelVertexBuffersLayout,
        },
        fragment: {
            module: terrainShaderModule,
            entryPoint: "fragment_main",
            targets: [{
                format: navigator.gpu.getPreferredCanvasFormat(),
            }],
        },
        primitive: {
            topology: "triangle-list",
            cullMode: 'back',
        },
        depthStencil: {
            depthWriteEnabled: true,
            depthCompare: 'less',
            format: 'depth24plus',
        },
    });

    const depthTexture = device.createTexture({
        size: [canvas.width, canvas.height],
        format: 'depth24plus',
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });

    const bindGroup = device.createBindGroup({
        layout: voxelPipeline.getBindGroupLayout(0),
        entries: [{
            binding: 0,
            resource: {
                buffer: uniformBuffer,
            }
        }]
    });

    // Setup
    const input = window.inputState || { deltaX: 0, deltaY: 0, velocityX: 0, velocityY: 0, zoom: 7, interacting: false, mouseX: 0, mouseY: 0, ndcX: 0, ndcY: 0 };
    
    // ---- Matrices ----
    const proj = Utils.mat4Create();
    const view = Utils.mat4Create();
    const mv = Utils.mat4Create();
    const mvp = Utils.mat4Create();
    const norm = Utils.mat4Create();

    // Persistent model rotation matrix — accumulated over time
    const modelRotation = Utils.mat4Create();
    const temp = Utils.mat4Create();

    // Apply an initial tilt so we see three faces
    Utils.mat4RotateX(modelRotation, modelRotation, 0.35);

    // Projection
    const aspect = canvas.width / canvas.height;
    Utils.mat4Perspective(proj, Math.PI / 6, aspect, 0.1, 100);
    
    const lightDirWorld = [0.0, 0.7, 1.0]; 

    function frame() {
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
            Utils.mat4Multiply(temp, inc, modelRotation);
            modelRotation.set(temp);
        }

        // Update view matrix with current zoom
        const cam_pos = [0, 0, input.zoom];
        Utils.mat4LookAt(view, cam_pos, [0, 0, 0], [0, 1, 0]);

        getMouse3D(cam_pos, input.ndcX, input.ndcY);

        // MVP
        Utils.mat4Multiply(mv, view, modelRotation);
        Utils.mat4Multiply(mvp, proj, mv);

        // Normal matrix (inverse-transpose of model-view)
        // Utils.mat4InverseTranspose(norm, mv);

        // Update uniform data
        const uniformData = new Float32Array(20); // mvp(16), lightDirWorld(3), padding(1)
        uniformData.set(mvp, 0);
        uniformData.set(lightDirWorld, 16);
        
        device.queue.writeBuffer(uniformBuffer, 0, uniformData);

        const commandEncoder = device.createCommandEncoder();
        const renderPassDescriptor = {
            colorAttachments: [{
                view: context.getCurrentTexture().createView(),
                clearValue: clearColor,
                loadOp: "clear",
                storeOp: "store",
            }],
            depthStencilAttachment: {
                view: depthTexture.createView(),
                depthClearValue: 1.0,
                depthLoadOp: 'clear',
                depthStoreOp: 'store',
            },
        };

        const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
        passEncoder.setPipeline(voxelPipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.setVertexBuffer(0, voxelVertexBuffer);
        passEncoder.setIndexBuffer(voxelIndicesBuffer, 'uint16');
        passEncoder.drawIndexed(voxelMesh.triangle_count);
        passEncoder.end();

        device.queue.submit([commandEncoder.finish()]);
        requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);
}

init();