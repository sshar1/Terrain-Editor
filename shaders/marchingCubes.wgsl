struct Uniforms {
    brushSize : f32,
    brushStrength : f32,
    resolution : f32
}

struct DrawIndirectArgs {
    vertexCount: atomic<u32>,
    instanceCount: u32,
    firstVertex: u32,
    firstInstance: u32,
}

@group(0) @binding(0) var<storage, read> grid : array<f32>;
@group(0) @binding(1) var<storage, read_write> vertexBuffer : array<f32>;
@group(0) @binding(2) var<storage, read_write> indirectArgs: DrawIndirectArgs;
@group(0) @binding(3) var<storage, read> caseTable: array<i32>;
@group(0) @binding(4) var<uniform> uniforms: Uniforms;

// Edge connection table
const edgeCorners = array<vec2<u32>, 12>(
    vec2<u32>(0u, 1u), // Edge 0
    vec2<u32>(1u, 2u), // Edge 1
    vec2<u32>(2u, 3u), // Edge 2
    vec2<u32>(3u, 0u), // Edge 3
    vec2<u32>(4u, 5u), // Edge 4
    vec2<u32>(5u, 6u), // Edge 5
    vec2<u32>(6u, 7u), // Edge 6
    vec2<u32>(7u, 4u), // Edge 7
    vec2<u32>(0u, 4u), // Edge 8
    vec2<u32>(1u, 5u), // Edge 9
    vec2<u32>(2u, 6u), // Edge 10
    vec2<u32>(3u, 7u)  // Edge 11
);

fn getGridValue(idx: vec3<u32>) -> f32 {
    let r = u32(uniforms.resolution);
    if (idx.x >= r || idx.y >= r || idx.z >= r) {
        return 0.0;
    }
    return grid[idx.x + (idx.y * r) + (idx.z * r * r)];
}

fn idx_to_pos(idx: vec3<u32>) -> vec3f {
    let r = uniforms.resolution - 1.0;  // so index 0 → -1.0 and index (res-1) → +1.0
    let x = 2.0 * (f32(idx.x) / r) - 1.0;
    let y = 2.0 * (f32(idx.y) / r) - 1.0;
    let z = 2.0 * (f32(idx.z) / r) - 1.0;
    return vec3f(x, y, z);
}

fn writeVertex(vertexIndex: u32, pos: vec3f, norm: vec3f, col: vec3f) {
    let offset = vertexIndex * 9u;
    vertexBuffer[offset + 0u] = pos.x;
    vertexBuffer[offset + 1u] = pos.y;
    vertexBuffer[offset + 2u] = pos.z;
    vertexBuffer[offset + 3u] = norm.x;
    vertexBuffer[offset + 4u] = norm.y;
    vertexBuffer[offset + 5u] = norm.z;
    vertexBuffer[offset + 6u] = col.x;
    vertexBuffer[offset + 7u] = col.y;
    vertexBuffer[offset + 8u] = col.z;
}

fn getEdgeVertex(edgeIndex: i32, cornerPos: array<vec3f, 8>, cornerVal: array<f32, 8>, iso: f32) -> vec3f {
    let corners = edgeCorners[edgeIndex];
    let idxA = corners.x;
    let idxB = corners.y;
    
    let posA = cornerPos[idxA];
    let posB = cornerPos[idxB];
    let valA = cornerVal[idxA];
    let valB = cornerVal[idxB];
    
    var mu = 0.5;
    let diff = valB - valA;
    if (abs(diff) > 0.00001) {
        mu = (iso - valA) / diff;
    }
    return mix(posA, posB, clamp(mu, 0.0, 1.0));
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let res = u32(uniforms.resolution);
    if (id.x >= res - 1u || id.y >= res - 1u || id.z >= res - 1u) {
        return;
    }

    let c0 = id + vec3<u32>(0u, 0u, 0u);
    let c1 = id + vec3<u32>(1u, 0u, 0u);
    let c2 = id + vec3<u32>(1u, 1u, 0u);
    let c3 = id + vec3<u32>(0u, 1u, 0u);
    let c4 = id + vec3<u32>(0u, 0u, 1u);
    let c5 = id + vec3<u32>(1u, 0u, 1u);
    let c6 = id + vec3<u32>(1u, 1u, 1u);
    let c7 = id + vec3<u32>(0u, 1u, 1u);

    let val0 = getGridValue(c0);
    let val1 = getGridValue(c1);
    let val2 = getGridValue(c2);
    let val3 = getGridValue(c3);
    let val4 = getGridValue(c4);
    let val5 = getGridValue(c5);
    let val6 = getGridValue(c6);
    let val7 = getGridValue(c7);

    var cubeIndex = 0u;
    let iso = 0.5;

    if (val0 >= iso) { cubeIndex |= 1u; }
    if (val1 >= iso) { cubeIndex |= 2u; }
    if (val2 >= iso) { cubeIndex |= 8u; }
    if (val3 >= iso) { cubeIndex |= 4u; }
    if (val4 >= iso) { cubeIndex |= 16u; }
    if (val5 >= iso) { cubeIndex |= 32u; }
    if (val6 >= iso) { cubeIndex |= 128u; }
    if (val7 >= iso) { cubeIndex |= 64u; }

    let tableOffset = cubeIndex * 16u;
    let firstEdge = caseTable[tableOffset + 0u];
    if (firstEdge == -1) {
        return;
    }

    let p0 = idx_to_pos(c0);
    let p1 = idx_to_pos(c1);
    let p2 = idx_to_pos(c2);
    let p3 = idx_to_pos(c3);
    let p4 = idx_to_pos(c4);
    let p5 = idx_to_pos(c5);
    let p6 = idx_to_pos(c6);
    let p7 = idx_to_pos(c7);

    let cornerPos = array<vec3f, 8>(p0, p1, p2, p3, p4, p5, p6, p7);
    let cornerVal = array<f32, 8>(val0, val1, val2, val3, val4, val5, val6, val7);

    for (var i = 0u; i < 15u; i += 3u) {
        let edge0 = caseTable[tableOffset + i];
        if (edge0 == -1) {
            break;
        }
        let edge1 = caseTable[tableOffset + i + 1u];
        let edge2 = caseTable[tableOffset + i + 2u];

        let startVertex = atomicAdd(&indirectArgs.vertexCount, 3u);
        // Prevent buffer overflow (500000 max vertices)
        if (startVertex + 3u > 500000u) {
            break;
        }

        let pos0 = getEdgeVertex(edge0, cornerPos, cornerVal, iso);
        let pos1 = getEdgeVertex(edge1, cornerPos, cornerVal, iso);
        let pos2 = getEdgeVertex(edge2, cornerPos, cornerVal, iso);

        // Compute normal pointing outward, with CCW winding (pos0, pos2, pos1)
        let normal = normalize(cross(pos2 - pos0, pos1 - pos0));

        let upFactor = normal.y;
        let grassColor = vec3f(0.3, 0.65, 0.3);
        let dirtColor = vec3f(0.5, 0.4, 0.3);
        let color = mix(dirtColor, grassColor, clamp((upFactor - 0.3) / 0.4, 0.0, 1.0));

        writeVertex(startVertex + 0u, pos0, normal, color);
        writeVertex(startVertex + 1u, pos2, normal, color);
        writeVertex(startVertex + 2u, pos1, normal, color);
    }
}