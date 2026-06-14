struct CursorData {
    cursorPosition : vec3f,
    cursorHit : u32
}

struct Uniforms {
    brushSize : f32,
    brushStrength : f32,
    resolution : f32
}

@group(0) @binding(0) var<storage, read_write> grid : array<f32>;
@group(0) @binding(1) var<storage, read> cursor : CursorData;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(8, 8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (cursor.cursorHit == 1) {
        let distFromHit = distance(cursor.cursorPosition, idx_to_pos(id));
        if (distFromHit < uniforms.brushSize) {
            // TODO maybe do a fall-off effect?
            grid[idxVecToFlat(id)] += uniforms.brushStrength;
        }
    }
}

fn idx_to_pos(idx: vec3<u32>) -> vec3f {
    let x = 2.0 * (f32(idx.x) / uniforms.resolution) - 1.0;
    let y = 2.0 * (f32(idx.y) / uniforms.resolution) - 1.0;
    let z = 2.0 * (f32(idx.z) / uniforms.resolution) - 1.0;
    return vec3f(x, y, z);
}

fn idxVecToFlat(idx: vec3<u32>) -> u32 {
    return idx.x + (idx.y * 32) + (idx.z * 32 * 32);
}