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

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (cursor.cursorHit == 1u) {
        let dist = distance(cursor.cursorPosition, idx_to_pos(id));
        if (dist < uniforms.brushSize) {
            // Smooth cubic falloff: 1.0 at centre → 0.0 at brushSize edge
            let t       = dist / uniforms.brushSize;          // [0, 1]
            let falloff = 1.0 - smoothstep(0.0, 1.0, t);     // cubic S-curve
            grid[idxVecToFlat(id)] += uniforms.brushStrength * falloff;
        }
    }
}

fn idx_to_pos(idx: vec3<u32>) -> vec3f {
    let r = uniforms.resolution - 1.0;  // matches marchingCubes.wgsl
    let x = 2.0 * (f32(idx.x) / r) - 1.0;
    let y = 2.0 * (f32(idx.y) / r) - 1.0;
    let z = 2.0 * (f32(idx.z) / r) - 1.0;
    return vec3f(x, y, z);
}

fn idxVecToFlat(idx: vec3<u32>) -> u32 {
    let r = u32(uniforms.resolution);
    return idx.x + (idx.y * r) + (idx.z * r * r);
}