struct CursorData {
    cursorPosition : vec3f,
    cursorHit : u32
}

// Layout (96 bytes total):
//   mvp          : mat4x4f  (bytes 0-63)
//   rayOrigin    : vec3f    (bytes 64-75)
//   resolution   : f32      (bytes 76-79)   <-- was _pad0
//   rayDir       : vec3f    (bytes 80-91)
//   sculptEnabled: f32      (bytes 92-95)   <-- was _pad1  (1.0 = sculpt on)
struct Uniforms {
    mvp : mat4x4f,
    rayOrigin : vec3f,
    resolution : f32,
    rayDir : vec3f,
    sculptEnabled : f32
}

@group(0) @binding(0) var<storage, read> grid : array<f32>;
@group(0) @binding(1) var<storage, read_write> cursor : CursorData;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

// Nearest-neighbour sample of the scalar field at model-space position pos.
fn sampleGrid(pos: vec3f) -> f32 {
    let res = uniforms.resolution;
    let gx = i32(floor((pos.x + 1.0) * 0.5 * res));
    let gy = i32(floor((pos.y + 1.0) * 0.5 * res));
    let gz = i32(floor((pos.z + 1.0) * 0.5 * res));
    let ir = i32(res);
    if (gx < 0 || gx >= ir || gy < 0 || gy >= ir || gz < 0 || gz >= ir) {
        return 0.0;
    }
    let ur = u32(res);
    return grid[u32(gx) + u32(gy) * ur + u32(gz) * ur * ur];
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    // Reset hit every frame
    cursor.cursorHit = 0u;

    // Only sculpt when Q is held
    if (uniforms.sculptEnabled < 0.5) {
        return;
    }

    let rayOrig = uniforms.rayOrigin;
    let rayDir  = uniforms.rayDir;
    let res     = uniforms.resolution;

    // --- Ray-AABB intersection with the [-1,1]^3 grid box ---
    // Guard against near-zero direction components
    var invDir: vec3f;
    invDir.x = select(1e10, 1.0 / rayDir.x, abs(rayDir.x) > 1e-6);
    invDir.y = select(1e10, 1.0 / rayDir.y, abs(rayDir.y) > 1e-6);
    invDir.z = select(1e10, 1.0 / rayDir.z, abs(rayDir.z) > 1e-6);

    let t0 = (-1.0 - rayOrig) * invDir;
    let t1 = ( 1.0 - rayOrig) * invDir;

    let tmin = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
    let tmax = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));

    // No intersection or box is entirely behind the ray
    if (tmax < 0.0 || tmin > tmax) {
        return;
    }

    // March from entry point in half-voxel steps
    let stepSize  = (2.0 / res) * 0.5;
    var t         = max(tmin, 0.0);
    var prevDens  = sampleGrid(rayOrig + rayDir * t);

    for (var i: i32 = 0; i < 200; i++) {
        t += stepSize;
        if (t > tmax) { break; }

        let pos  = rayOrig + rayDir * t;
        let dens = sampleGrid(pos);

        // Detect outside-to-inside crossing at the iso-surface (0.5)
        if (prevDens < 0.5 && dens >= 0.5) {
            // Refine to midpoint between the two samples
            cursor.cursorPosition = pos - rayDir * (stepSize * 0.5);
            cursor.cursorHit = 1u;
            return;
        }

        prevDens = dens;
    }
}