import * as Utils from './math_util.js'

const gridWidth = 100;
const numPoints = gridWidth * gridWidth;
const clearColor = { r: 0.03, g: 0.03, b: 0.05, a: 0.0 };

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
    const boxShaderModule = await createShaderModule(device, "./box.wgsl");

    // Uniforms: mat4x4 (64) + mat4x4 (64) + vec3 (12) + pad (4) = 144
    const boxUniformBufferSize = 144;
    const uniformBuffer = device.createBuffer({
        size: boxUniformBufferSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const box = createBox();
    const boxVertexBuffer = device.createBuffer({
        size: box.vertexData.byteLength,
        usage: GPUBufferUsage.VERTEX |  GPUBufferUsage.COPY_DST,
    });
    const boxIndicesBuffer = device.createBuffer({
        size: box.line_indices.byteLength,
        usage: GPUBufferUsage.INDEX |  GPUBufferUsage.COPY_DST,
    });

    device.queue.writeBuffer(boxVertexBuffer, 0, box.vertexData);
    device.queue.writeBuffer(boxIndicesBuffer, 0, box.line_indices);

    const boxVertexBuffersLayout = [{
        attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x3' },  // position
            { shaderLocation: 1, offset: 12, format: 'float32x3' }, // color
            { shaderLocation: 2, offset: 24, format: 'float32x3' }  // normal
        ],
        arrayStride: 36, // 9 floats * 4 bytes
        stepMode: 'vertex'
    }];

    const boxPipeline = device.createRenderPipeline({
        layout: "auto",
        vertex: {
            module: boxShaderModule,
            entryPoint: "vertex_main",
            buffers: boxVertexBuffersLayout,
        },
        fragment: {
            module: boxShaderModule,
            entryPoint: "fragment_main",
            targets: [{
                format: navigator.gpu.getPreferredCanvasFormat(),
            }],
        },
        primitive: {
            topology: "line-list",
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
        layout: boxPipeline.getBindGroupLayout(0),
        entries: [{
            binding: 0,
            resource: {
                buffer: uniformBuffer,
            }
        }]
    });

    // Setup
    const input = window.inputState || { deltaX: 0, deltaY: 0, velocityX: 0, velocityY: 0, zoom: 7, interacting: false };
    
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

        // MVP
        Utils.mat4Multiply(mv, view, modelRotation);
        Utils.mat4Multiply(mvp, proj, mv);

        // Normal matrix (inverse-transpose of model-view)
        Utils.mat4InverseTranspose(norm, mv);

        // Update uniform data
        const uniformData = new Float32Array(16 + 16 + 4); // mvp(16), norm(16), lightDir(3) + pad(1)
        uniformData.set(mvp, 0);
        uniformData.set(norm, 16);
        uniformData.set(lightDirWorld, 32);
        
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
        passEncoder.setPipeline(boxPipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.setVertexBuffer(0, boxVertexBuffer);
        passEncoder.setIndexBuffer(boxIndicesBuffer, 'uint16');
        passEncoder.drawIndexed(box.line_count);
        passEncoder.end();

        device.queue.submit([commandEncoder.finish()]);
        requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);
}

init();