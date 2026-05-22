struct VertexIn {
    @location(0) position : vec3f,
    @location(1) color : vec3f,
    @location(2) normal : vec3f
}

struct VertexOut {
    @builtin(position) position : vec4f,
    @location(0) vDot : f32,
    @location(1) vColor : vec3f
}

struct Uniforms {
    mvp : mat4x4f,
    normalMatrix : mat4x4f,
    lightDir : vec3f,
    _pad : f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@vertex
fn vertex_main(in: VertexIn) -> VertexOut
{
    let transNormal = normalize((uniforms.normalMatrix * vec4f(in.normal, 0.0)).xyz);

    var output : VertexOut;
    output.position = uniforms.mvp * vec4f(in.position, 1.0f);
    output.vDot = max(dot(transNormal, normalize(uniforms.lightDir)), 0.0);
    output.vColor = in.color;

    return output;
}

@fragment
fn fragment_main(fragData: VertexOut) -> @location(0) vec4f
{
    return vec4f(fragData.vColor, 1.0);
}