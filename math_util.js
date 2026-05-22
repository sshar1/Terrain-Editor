// ============================================================
//  Minimal matrix math (column-major, Float32Array)
// ============================================================

export function cross_product(a, b) {
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
}

export function dot(a, b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function normalize(v) {
    let len = Math.hypot(v[0], v[1], v[2]);
    if (len) { len = 1 / len; v[0] *= len; v[1] *= len; v[2] *= len; }
    return v;
}

export function mat4Create() {
    const m = new Float32Array(16);
    m[0] = m[5] = m[10] = m[15] = 1;
    return m;
}

export function mat4Multiply(out, a, b) {
    for (let i = 0; i < 4; i++) {
        const a0 = a[i], a4 = a[i + 4], a8 = a[i + 8], a12 = a[i + 12];
        out[i] = a0 * b[0] + a4 * b[1] + a8 * b[2] + a12 * b[3];
        out[i + 4] = a0 * b[4] + a4 * b[5] + a8 * b[6] + a12 * b[7];
        out[i + 8] = a0 * b[8] + a4 * b[9] + a8 * b[10] + a12 * b[11];
        out[i + 12] = a0 * b[12] + a4 * b[13] + a8 * b[14] + a12 * b[15];
    }
    return out;
}

// Returns the perspective projection matrix
export function mat4Perspective(out, fovY, aspect, near, far) {
    const f = 1.0 / Math.tan(fovY / 2);
    const nf = 1 / (near - far);
    out.fill(0);
    out[0] = f / aspect;
    out[5] = f;
    out[10] = (far + near) * nf;
    out[11] = -1;
    out[14] = 2 * far * near * nf;
    return out;
}

// Returns the view matrix.
// eye: position of the camera
// center: position of the center of the scene
// up: up direction of the camera
export function mat4LookAt(out, eye, center, up) {
    let x, y, z;

    z = normalize([eye[0] - center[0], eye[1] - center[1], eye[2] - center[2]]);
    x = normalize(cross_product(up, z));
    y = normalize(cross_product(z, x));

    out[0] = x[0]; out[1] = y[0]; out[2] = z[0]; out[3] = 0;
    out[4] = x[1]; out[5] = y[1]; out[6] = z[1]; out[7] = 0;
    out[8] = x[2]; out[9] = y[2]; out[10] = z[2]; out[11] = 0;
    out[12] = -dot(x, eye);
    out[13] = -dot(y, eye);
    out[14] = -dot(z, eye);
    out[15] = 1;
    return out;
}

export function mat4RotateX(out, a, rad) {
    const s = Math.sin(rad), c = Math.cos(rad);
    const t = mat4Create();
    t[5] = c; t[6] = s; t[9] = -s; t[10] = c;
    return mat4Multiply(out, a, t);
}

export function mat4RotateY(out, a, rad) {
    const s = Math.sin(rad), c = Math.cos(rad);
    const t = mat4Create();
    t[0] = c; t[2] = -s; t[8] = s; t[10] = c;
    return mat4Multiply(out, a, t);
}

export function mat4InverseTranspose(out, m) {
    // Extract the 3×3 upper-left of column-major matrix m.
    // m[col*4 + row], so mRC = m at row R, column C.
    const m00 = m[0], m10 = m[1], m20 = m[2];   // column 0
    const m01 = m[4], m11 = m[5], m21 = m[6];   // column 1
    const m02 = m[8], m12 = m[9], m22 = m[10];  // column 2

    const det = m00 * (m11 * m22 - m12 * m21)
        - m01 * (m10 * m22 - m12 * m20)
        + m02 * (m10 * m21 - m11 * m20);
    const id = 1.0 / det;

    // The inverse-transpose = cofactor matrix / determinant.
    // Store in column-major: out[col*4 + row] = Cofactor[row][col] / det
    out.fill(0);
    out[0] = (m11 * m22 - m12 * m21) * id;
    out[1] = -(m01 * m22 - m02 * m21) * id;
    out[2] = (m01 * m12 - m02 * m11) * id;
    out[4] = -(m10 * m22 - m12 * m20) * id;
    out[5] = (m00 * m22 - m02 * m20) * id;
    out[6] = -(m00 * m12 - m02 * m10) * id;
    out[8] = (m10 * m21 - m11 * m20) * id;
    out[9] = -(m00 * m21 - m01 * m20) * id;
    out[10] = (m00 * m11 - m01 * m10) * id;
    out[15] = 1;
    return out;
}

export function mat4Transpose(out, m) {
    out[0] = m[0]; out[1] = m[4]; out[2] = m[8]; out[3] = m[12];
    out[4] = m[1]; out[5] = m[5]; out[6] = m[9]; out[7] = m[13];
    out[8] = m[2]; out[9] = m[6]; out[10] = m[10]; out[11] = m[14];
    out[12] = m[3]; out[13] = m[7]; out[14] = m[11]; out[15] = m[15];
    return out;
}