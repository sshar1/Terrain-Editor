struct CursorData {
    cursorPosition : vec3f,
    cursorHit : u32
}

struct Uniforms {
    mvp : mat4x4f,
    rayOrigin : vec3f,
    _pad0 : f32,
    rayDir : vec3f,
    _pad1 : f32
}

@group(0) @binding(0) var<storage, read_write> grid : array<f32>;
@group(0) @binding(1) var<storage, read_write> cursor : CursorData;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    cursor.cursorHit = 1;
    cursor.cursorPosition = vec3f(0.0, 0.0, 0.0);
}

fn idx_to_pos(idx: vec3<u32>) -> vec3f {
    let x = 2.0 * (f32(idx.x) / 32.0) - 1.0;
    let y = 2.0 * (f32(idx.y) / 32.0) - 1.0;
    let z = 2.0 * (f32(idx.z) / 32.0) - 1.0;
    return vec3f(x, y, z);
}

fn idxVecToFlat(idx: vec3<u32>) -> u32 {
    return idx.x + (idx.y * 32) + (idx.z * 32 * 32);
}