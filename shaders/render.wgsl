struct VertexIn {
    @location(0) position : vec3f,
    @location(1) normal : vec3f,
    @location(2) color : vec3f
}

struct VertexOut {
    @builtin(position) position : vec4f,
    @location(0) vDot : f32,
    @location(1) color : vec3f
}

struct Uniforms {
    mvp : mat4x4f,
    lightDir : vec3f,
    _pad0 : f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@vertex
fn vertex_main(in: VertexIn) -> VertexOut
{
    let transNormal = normalize((uniforms.mvp * vec4f(in.normal, 0.0)).xyz);

    var output : VertexOut;
    output.position = uniforms.mvp * vec4f(in.position, 1.0f);
    output.vDot = max(dot(transNormal, normalize(uniforms.lightDir)), 0.0);
    output.color = in.color;

    return output;
}

@fragment
fn fragment_main(fragData: VertexOut) -> @location(0) vec4f
{
    var ambient = 0.2;
    var lighting = ambient + (1 - ambient) * fragData.vDot;

    return vec4f(fragData.color, 1.0) * lighting;
}