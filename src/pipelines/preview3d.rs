//! Depth-tested 3D tool preview pipeline (points, lines, line strips, opaque & translucent triangles)
//! with D-P64-10 multi-pass front-to-back depth peeling (`K = 4`), `Rgba16Float` 2-MRT tail,
//! and static 64x64 Void-and-Cluster blue-noise importance-sampled single-depth resolve.

use glam::{Mat4, Vec4};

use crate::camera::RawCamera;
use crate::pipelines::blue_noise::{BLUE_NOISE_64X64, BLUE_NOISE_DIM};
use crate::pipelines::DynamicGpuBuffer;

const PEEL_LAYER_COUNT: usize = 4;

const PREVIEW3D_WGSL: &str = r#"
struct CameraUniform {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct PeelPassUniform {
    pass_idx: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(1) @binding(0)
var opaque_depth_tex: texture_depth_2d;

@group(1) @binding(1)
var prev_depth_tex: texture_depth_2d;

@group(1) @binding(2)
var<uniform> peel_params: PeelPassUniform;

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) color: vec4<f32>,
};

struct VsOut {
    @invariant @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip_pos = camera.view_proj * vec4<f32>(in.pos, 1.0);
    out.color = in.color;
    return out;
}

@vertex
fn vs_depth_stamp(in: VsIn) -> VsOut {
    var out: VsOut;
    var clip = camera.view_proj * vec4<f32>(in.pos, 1.0);
    clip.z = 0.0;
    out.clip_pos = clip;
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return in.color;
}

@fragment
fn fs_depth_stamp(in: VsOut) -> @location(0) vec4<f32> {
    if (in.color.a <= 1e-6) {
        discard;
    }
    return in.color;
}

@fragment
fn fs_peel(in: VsOut) -> @location(0) vec4<f32> {
    if (in.color.a <= 1e-5) {
        discard;
    }
    let px = vec2<i32>(in.clip_pos.xy);
    let z_opaque = textureLoad(opaque_depth_tex, px, 0);
    if (in.clip_pos.z > z_opaque) {
        discard;
    }
    if (peel_params.pass_idx > 0u) {
        let z_prev = textureLoad(prev_depth_tex, px, 0);
        if (in.clip_pos.z <= z_prev) {
            discard;
        }
    }
    return in.color;
}

struct TailOut {
    @location(0) accum: vec4<f32>,
    @location(1) tail_meta: vec4<f32>,
};

@fragment
fn fs_tail(in: VsOut) -> TailOut {
    let a = in.color.a;
    if (a <= 1e-5) {
        discard;
    }
    let px = vec2<i32>(in.clip_pos.xy);
    let z_opaque = textureLoad(opaque_depth_tex, px, 0);
    let z_peel3 = textureLoad(prev_depth_tex, px, 0);
    if (in.clip_pos.z > z_opaque || in.clip_pos.z <= z_peel3) {
        discard;
    }
    var out: TailOut;
    out.accum = vec4<f32>(in.color.rgb * a, a);
    out.tail_meta = vec4<f32>(a, a, a, 1.0 - in.clip_pos.z);
    return out;
}
"#;

const RESOLVE_WGSL: &str = r#"
struct ResolveVsOut {
    @builtin(position) pos: vec4<f32>,
};

@vertex
fn vs_resolve(@builtin(vertex_index) vi: u32) -> ResolveVsOut {
    let uv = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    var out: ResolveVsOut;
    out.pos = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    return out;
}

@group(0) @binding(0) var opaque_color_tex: texture_2d<f32>;
@group(0) @binding(1) var opaque_depth_tex: texture_depth_2d;
@group(0) @binding(2) var peel_color_0: texture_2d<f32>;
@group(0) @binding(3) var peel_depth_0: texture_depth_2d;
@group(0) @binding(4) var peel_color_1: texture_2d<f32>;
@group(0) @binding(5) var peel_depth_1: texture_depth_2d;
@group(0) @binding(6) var peel_color_2: texture_2d<f32>;
@group(0) @binding(7) var peel_depth_2: texture_depth_2d;
@group(0) @binding(8) var peel_color_3: texture_2d<f32>;
@group(0) @binding(9) var peel_depth_3: texture_depth_2d;
@group(0) @binding(10) var tail_accum_tex: texture_2d<f32>;
@group(0) @binding(11) var tail_meta_tex: texture_2d<f32>;
@group(0) @binding(12) var blue_noise_tex: texture_2d<u32>;

struct ResolveFsOut {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

@fragment
fn fs_resolve(in: ResolveVsOut) -> ResolveFsOut {
    let px = vec2<i32>(in.pos.xy);

    var layers_rgb: array<vec3<f32>, 6>;
    var layers_w: array<f32, 6>;
    var layers_z: array<f32, 6>;
    var count: u32 = 0u;
    var transmittance: f32 = 1.0;
    var total_alpha: f32 = 0.0;

    let z0 = textureLoad(peel_depth_0, px, 0);
    if (z0 < 1.0) {
        let c0 = textureLoad(peel_color_0, px, 0);
        if (c0.a > 1e-5) {
            let w = transmittance * c0.a;
            layers_rgb[count] = c0.rgb;
            layers_w[count] = w;
            layers_z[count] = z0;
            count = count + 1u;
            total_alpha = total_alpha + w;
            transmittance = transmittance * (1.0 - c0.a);
        }
    }

    let z1 = textureLoad(peel_depth_1, px, 0);
    if (z1 < 1.0) {
        let c1 = textureLoad(peel_color_1, px, 0);
        if (c1.a > 1e-5) {
            let w = transmittance * c1.a;
            layers_rgb[count] = c1.rgb;
            layers_w[count] = w;
            layers_z[count] = z1;
            count = count + 1u;
            total_alpha = total_alpha + w;
            transmittance = transmittance * (1.0 - c1.a);
        }
    }

    let z2 = textureLoad(peel_depth_2, px, 0);
    if (z2 < 1.0) {
        let c2 = textureLoad(peel_color_2, px, 0);
        if (c2.a > 1e-5) {
            let w = transmittance * c2.a;
            layers_rgb[count] = c2.rgb;
            layers_w[count] = w;
            layers_z[count] = z2;
            count = count + 1u;
            total_alpha = total_alpha + w;
            transmittance = transmittance * (1.0 - c2.a);
        }
    }

    let z3 = textureLoad(peel_depth_3, px, 0);
    if (z3 < 1.0) {
        let c3 = textureLoad(peel_color_3, px, 0);
        if (c3.a > 1e-5) {
            let w = transmittance * c3.a;
            layers_rgb[count] = c3.rgb;
            layers_w[count] = w;
            layers_z[count] = z3;
            count = count + 1u;
            total_alpha = total_alpha + w;
            transmittance = transmittance * (1.0 - c3.a);
        }
    }

    let tail_accum = textureLoad(tail_accum_tex, px, 0);
    let tail_meta = textureLoad(tail_meta_tex, px, 0);
    if (tail_accum.a > 1e-5 && tail_meta.a > 0.0) {
        let a_tail = clamp(1.0 - tail_meta.r, 0.0, 1.0);
        if (a_tail > 1e-5) {
            let c_tail = clamp(tail_accum.rgb / tail_accum.a, vec3<f32>(0.0), vec3<f32>(1.0));
            let z_tail = clamp(1.0 - tail_meta.a, 0.0, 0.9999999);
            let w = transmittance * a_tail;
            layers_rgb[count] = c_tail;
            layers_w[count] = w;
            layers_z[count] = z_tail;
            count = count + 1u;
            total_alpha = total_alpha + w;
            transmittance = transmittance * (1.0 - a_tail);
        }
    }

    let z_op = textureLoad(opaque_depth_tex, px, 0);
    if (z_op < 1.0) {
        let op = textureLoad(opaque_color_tex, px, 0);
        if (op.a > 1e-5) {
            let c_op = clamp(op.rgb / op.a, vec3<f32>(0.0), vec3<f32>(1.0));
            let w = transmittance * op.a;
            layers_rgb[count] = c_op;
            layers_w[count] = w;
            layers_z[count] = z_op;
            count = count + 1u;
            total_alpha = total_alpha + w;
            transmittance = transmittance * (1.0 - op.a);
        }
    }

    var out: ResolveFsOut;
    if (count == 0u || total_alpha <= 1e-5) {
        out.color = vec4<f32>(0.0);
        out.depth = 1.0;
        return out;
    }

    let final_alpha = clamp(total_alpha, 0.0, 1.0);
    if (count == 1u) {
        out.color = vec4<f32>(layers_rgb[0] * final_alpha, final_alpha);
        out.depth = layers_z[0];
        return out;
    }

    let bn_coord = vec2<i32>(vec2<u32>(px) & vec2<u32>(63u));
    let bn = textureLoad(blue_noise_tex, bn_coord, 0).r;
    let u = (f32(bn) + 0.5) / 256.0;
    let cdf_threshold = u * total_alpha;

    var acc: f32 = 0.0;
    var sel: u32 = count - 1u;
    for (var i: u32 = 0u; i < count; i = i + 1u) {
        acc = acc + layers_w[i];
        if (cdf_threshold <= acc) {
            sel = i;
            break;
        }
    }

    out.color = vec4<f32>(layers_rgb[sel] * final_alpha, final_alpha);
    out.depth = layers_z[sel];
    return out;
}
"#;

/// Vertex format for 3D tool preview primitives (position + straight RGBA color).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PreviewVertex3D {
    pub position: [f32; 3],
    pub color: [f32; 4],
}

impl PreviewVertex3D {
    pub const fn new(position: [f32; 3], color: [f32; 4]) -> Self {
        Self { position, color }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct PreviewCameraUniform {
    view_proj: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct PeelPassUniform {
    pass_idx: u32,
    _pad: [u32; 3],
}

/// Geometry slices submitted to a single [`Preview3dPipeline`] render pass.
#[derive(Clone, Copy, Debug, Default)]
pub struct Preview3dPassData<'a> {
    pub points: &'a [PreviewVertex3D],
    pub lines: &'a [PreviewVertex3D],
    pub line_strips: &'a [&'a [PreviewVertex3D]],
    pub opaque_triangles: &'a [PreviewVertex3D],
    pub opaque_indices: Option<&'a [u32]>,
    pub translucent_triangles: &'a [PreviewVertex3D],
    pub translucent_indices: Option<&'a [u32]>,
}

impl Preview3dPassData<'_> {
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
            && self.lines.is_empty()
            && self.line_strips.iter().all(|s| s.len() < 2)
            && self.opaque_triangles.is_empty()
            && self.translucent_triangles.is_empty()
    }

    pub fn has_translucent(&self) -> bool {
        !self.translucent_triangles.is_empty()
            && self.translucent_indices.map_or(true, |idx| !idx.is_empty())
    }
}

#[derive(Clone, Copy, Debug)]
enum PreviewPipelineKind {
    Points,
    Lines,
    LineStrip,
    OpaqueTriangles,
    TranslucentTriangles,
}

#[derive(Clone, Copy, Debug)]
struct StagedDraw {
    kind: PreviewPipelineKind,
    vertex_start: u32,
    vertex_count: u32,
    index_start: u32,
    index_count: u32,
    indexed: bool,
}

/// Depth behavior for [`Preview3dPipeline`] draw calls.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Preview3dDepthMode {
    /// Standard depth-tested rendering (`LessEqual`); translucent triangles do not write depth.
    #[default]
    Standard,
    /// Depth-tested rendering (`LessEqual`) where translucent triangles also write real WGPU NDC depth.
    DepthTestedTranslucentWrite,
    /// Always-on-top depth-stamp rendering (`Always`, `depth_write = true`, `clip_pos.z = 0.0`).
    DepthStamp,
}

struct DepthPeelScratch {
    width: u32,
    height: u32,
    _opaque_color_tex: wgpu::Texture,
    opaque_color_view: wgpu::TextureView,
    _opaque_depth_tex: wgpu::Texture,
    opaque_depth_view: wgpu::TextureView,
    _peel_color_tex: [wgpu::Texture; PEEL_LAYER_COUNT],
    peel_color_views: [wgpu::TextureView; PEEL_LAYER_COUNT],
    _peel_depth_tex: [wgpu::Texture; PEEL_LAYER_COUNT],
    peel_depth_views: [wgpu::TextureView; PEEL_LAYER_COUNT],
    _tail_accum_tex: wgpu::Texture,
    tail_accum_view: wgpu::TextureView,
    _tail_meta_tex: wgpu::Texture,
    tail_meta_view: wgpu::TextureView,
    peel_bind_groups: [wgpu::BindGroup; PEEL_LAYER_COUNT],
    tail_bind_group: wgpu::BindGroup,
    resolve_bind_group: wgpu::BindGroup,
}

struct DepthPeelPipelines {
    color_format: wgpu::TextureFormat,
    depth_format: wgpu::TextureFormat,
    peel_bgl: wgpu::BindGroupLayout,
    tail_bgl: wgpu::BindGroupLayout,
    resolve_bgl: wgpu::BindGroupLayout,
    peel_pipeline: wgpu::RenderPipeline,
    tail_pipeline: wgpu::RenderPipeline,
    resolve_pipeline: wgpu::RenderPipeline,
    peel_pass_uniforms: [wgpu::Buffer; PEEL_LAYER_COUNT],
    blue_noise_texture: wgpu::Texture,
    blue_noise_view: wgpu::TextureView,
    blue_noise_uploaded: bool,
    scratch: Option<DepthPeelScratch>,
}

fn preview_vertex_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<PreviewVertex3D>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 12,
                shader_location: 1,
            },
        ],
    }
}

fn depth_tex_bgl_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Depth,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn float_tex_bgl_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn uint_tex_bgl_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Uint,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn create_2d_scratch_texture(
    device: &wgpu::Device,
    label: &str,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

impl DepthPeelPipelines {
    fn new(
        device: &wgpu::Device,
        camera_bgl: &wgpu::BindGroupLayout,
        preview_shader: &wgpu::ShaderModule,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
    ) -> Self {
        let resolve_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tricore_preview3d_peel_resolve_shader"),
            source: wgpu::ShaderSource::Wgsl(RESOLVE_WGSL.into()),
        });

        let peel_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tricore_preview3d_peel_bgl"),
            entries: &[
                depth_tex_bgl_entry(0),
                depth_tex_bgl_entry(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let tail_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tricore_preview3d_tail_bgl"),
            entries: &[depth_tex_bgl_entry(0), depth_tex_bgl_entry(1)],
        });

        let resolve_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tricore_preview3d_resolve_bgl"),
            entries: &[
                float_tex_bgl_entry(0),
                depth_tex_bgl_entry(1),
                float_tex_bgl_entry(2),
                depth_tex_bgl_entry(3),
                float_tex_bgl_entry(4),
                depth_tex_bgl_entry(5),
                float_tex_bgl_entry(6),
                depth_tex_bgl_entry(7),
                float_tex_bgl_entry(8),
                depth_tex_bgl_entry(9),
                float_tex_bgl_entry(10),
                float_tex_bgl_entry(11),
                uint_tex_bgl_entry(12),
            ],
        });

        let peel_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tricore_preview3d_peel_layout"),
            bind_group_layouts: &[Some(camera_bgl), Some(&peel_bgl)],
            immediate_size: 0,
        });
        let tail_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tricore_preview3d_tail_layout"),
            bind_group_layouts: &[Some(camera_bgl), Some(&tail_bgl)],
            immediate_size: 0,
        });
        let resolve_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tricore_preview3d_resolve_layout"),
            bind_group_layouts: &[Some(&resolve_bgl)],
            immediate_size: 0,
        });

        let peel_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tricore_preview3d_peel_pipeline"),
            layout: Some(&peel_layout),
            vertex: wgpu::VertexState {
                module: preview_shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[Some(preview_vertex_layout())],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: preview_shader,
                entry_point: Some("fs_peel"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let tail_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tricore_preview3d_tail_pipeline"),
            layout: Some(&tail_layout),
            vertex: wgpu::VertexState {
                module: preview_shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[Some(preview_vertex_layout())],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: preview_shader,
                entry_point: Some("fs_tail"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::Zero,
                                dst_factor: wgpu::BlendFactor::OneMinusSrc,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Max,
                            },
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
            }),
            multiview_mask: None,
            cache: None,
        });

        let resolve_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tricore_preview3d_resolve_pipeline"),
            layout: Some(&resolve_layout),
            vertex: wgpu::VertexState {
                module: &resolve_shader,
                entry_point: Some("vs_resolve"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &resolve_shader,
                entry_point: Some("fs_resolve"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let peel_pass_uniforms = std::array::from_fn(|i| {
            let uniform = PeelPassUniform {
                pass_idx: i as u32,
                _pad: [0; 3],
            };
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("tricore_preview3d_peel_pass_uniform"),
                size: std::mem::size_of::<PeelPassUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM,
                mapped_at_creation: true,
            });
            buf.slice(..)
                .get_mapped_range_mut()
                .expect("peel pass uniform mapped at creation")
                .copy_from_slice(bytemuck::bytes_of(&uniform));
            buf.unmap();
            buf
        });

        let blue_noise_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tricore_preview3d_blue_noise_64x64"),
            size: wgpu::Extent3d {
                width: BLUE_NOISE_DIM,
                height: BLUE_NOISE_DIM,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let blue_noise_view =
            blue_noise_texture.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            color_format,
            depth_format,
            peel_bgl,
            tail_bgl,
            resolve_bgl,
            peel_pipeline,
            tail_pipeline,
            resolve_pipeline,
            peel_pass_uniforms,
            blue_noise_texture,
            blue_noise_view,
            blue_noise_uploaded: false,
            scratch: None,
        }
    }

    fn ensure_blue_noise_uploaded(&mut self, queue: &wgpu::Queue) {
        if self.blue_noise_uploaded {
            return;
        }
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.blue_noise_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &BLUE_NOISE_64X64,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(BLUE_NOISE_DIM),
                rows_per_image: Some(BLUE_NOISE_DIM),
            },
            wgpu::Extent3d {
                width: BLUE_NOISE_DIM,
                height: BLUE_NOISE_DIM,
                depth_or_array_layers: 1,
            },
        );
        self.blue_noise_uploaded = true;
    }

    fn ensure_scratch(&mut self, device: &wgpu::Device, width: u32, height: u32) -> &DepthPeelScratch {
        let w = width.max(1);
        let h = height.max(1);
        let needs_alloc = self
            .scratch
            .as_ref()
            .is_none_or(|s| s.width != w || s.height != h);
        if needs_alloc {
            let (opaque_color_tex, opaque_color_view) = create_2d_scratch_texture(
                device,
                "tricore_peel_opaque_color",
                w,
                h,
                self.color_format,
            );
            let (opaque_depth_tex, opaque_depth_view) = create_2d_scratch_texture(
                device,
                "tricore_peel_opaque_depth",
                w,
                h,
                self.depth_format,
            );

            let peel_colors = std::array::from_fn::<_, PEEL_LAYER_COUNT, _>(|_| {
                create_2d_scratch_texture(
                    device,
                    "tricore_peel_layer_color",
                    w,
                    h,
                    wgpu::TextureFormat::Rgba8Unorm,
                )
            });
            let [pc0, pc1, pc2, pc3] = peel_colors;
            let peel_color_tex = [pc0.0, pc1.0, pc2.0, pc3.0];
            let peel_color_views = [pc0.1, pc1.1, pc2.1, pc3.1];

            let peel_depths = std::array::from_fn::<_, PEEL_LAYER_COUNT, _>(|_| {
                create_2d_scratch_texture(
                    device,
                    "tricore_peel_layer_depth",
                    w,
                    h,
                    self.depth_format,
                )
            });
            let [pd0, pd1, pd2, pd3] = peel_depths;
            let peel_depth_tex = [pd0.0, pd1.0, pd2.0, pd3.0];
            let peel_depth_views = [pd0.1, pd1.1, pd2.1, pd3.1];

            let (tail_accum_tex, tail_accum_view) = create_2d_scratch_texture(
                device,
                "tricore_peel_tail_accum",
                w,
                h,
                wgpu::TextureFormat::Rgba16Float,
            );
            let (tail_meta_tex, tail_meta_view) = create_2d_scratch_texture(
                device,
                "tricore_peel_tail_meta",
                w,
                h,
                wgpu::TextureFormat::Rgba16Float,
            );

            let peel_bind_groups = std::array::from_fn(|p| {
                let prev_view = if p == 0 {
                    &opaque_depth_view
                } else {
                    &peel_depth_views[p - 1]
                };
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("tricore_preview3d_peel_bg"),
                    layout: &self.peel_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&opaque_depth_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(prev_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.peel_pass_uniforms[p].as_entire_binding(),
                        },
                    ],
                })
            });

            let tail_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tricore_preview3d_tail_bg"),
                layout: &self.tail_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&opaque_depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            &peel_depth_views[PEEL_LAYER_COUNT - 1],
                        ),
                    },
                ],
            });

            let resolve_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tricore_preview3d_resolve_bg"),
                layout: &self.resolve_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&opaque_color_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&opaque_depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&peel_color_views[0]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&peel_depth_views[0]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&peel_color_views[1]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(&peel_depth_views[1]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::TextureView(&peel_color_views[2]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::TextureView(&peel_depth_views[2]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: wgpu::BindingResource::TextureView(&peel_color_views[3]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: wgpu::BindingResource::TextureView(&peel_depth_views[3]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: wgpu::BindingResource::TextureView(&tail_accum_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 11,
                        resource: wgpu::BindingResource::TextureView(&tail_meta_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 12,
                        resource: wgpu::BindingResource::TextureView(&self.blue_noise_view),
                    },
                ],
            });

            self.scratch = Some(DepthPeelScratch {
                width: w,
                height: h,
                _opaque_color_tex: opaque_color_tex,
                opaque_color_view,
                _opaque_depth_tex: opaque_depth_tex,
                opaque_depth_view,
                _peel_color_tex: peel_color_tex,
                peel_color_views,
                _peel_depth_tex: peel_depth_tex,
                peel_depth_views,
                _tail_accum_tex: tail_accum_tex,
                tail_accum_view,
                _tail_meta_tex: tail_meta_tex,
                tail_meta_view,
                peel_bind_groups,
                tail_bind_group,
                resolve_bind_group,
            });
        }
        self.scratch.as_ref().expect("scratch allocated")
    }
}

/// Computes a conservative screen-space scissor rect `[x, y, w, h]` for the world-space AABB
/// of `verts` with a 1px safety margin. Falls back to the full viewport `[0, 0, phys_w, phys_h]`
/// if any AABB corner has `clip.w <= 1e-6` or non-finite coordinates.
fn compute_translucent_scissor_rect(
    verts: &[PreviewVertex3D],
    view_proj: Mat4,
    phys_w: u32,
    phys_h: u32,
) -> [u32; 4] {
    let full = [0, 0, phys_w, phys_h];
    if verts.is_empty() || phys_w == 0 || phys_h == 0 {
        return [0, 0, 0, 0];
    }

    let mut min_p = [f32::INFINITY; 3];
    let mut max_p = [f32::NEG_INFINITY; 3];
    for v in verts {
        for axis in 0..3 {
            let c = v.position[axis];
            if !c.is_finite() {
                return full;
            }
            min_p[axis] = min_p[axis].min(c);
            max_p[axis] = max_p[axis].max(c);
        }
    }

    let xs = [min_p[0], max_p[0]];
    let ys = [min_p[1], max_p[1]];
    let zs = [min_p[2], max_p[2]];

    let wf = phys_w as f32;
    let hf = phys_h as f32;
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;

    for &x in &xs {
        for &y in &ys {
            for &z in &zs {
                let clip = view_proj * Vec4::new(x, y, z, 1.0);
                if !clip.x.is_finite()
                    || !clip.y.is_finite()
                    || !clip.w.is_finite()
                    || clip.w <= 1e-6
                {
                    return full;
                }
                let ndc_x = clip.x / clip.w;
                let ndc_y = clip.y / clip.w;
                let px_x = (ndc_x * 0.5 + 0.5) * wf;
                let px_y = (0.5 - ndc_y * 0.5) * hf;
                min_x = min_x.min(px_x);
                max_x = max_x.max(px_x);
                min_y = min_y.min(px_y);
                max_y = max_y.max(px_y);
            }
        }
    }

    let x0 = (min_x.floor() - 1.0).clamp(0.0, wf) as u32;
    let y0 = (min_y.floor() - 1.0).clamp(0.0, hf) as u32;
    let x1 = (max_x.ceil() + 1.0).clamp(0.0, wf) as u32;
    let y1 = (max_y.ceil() + 1.0).clamp(0.0, hf) as u32;
    if x1 <= x0 || y1 <= y0 {
        return [0, 0, 0, 0];
    }
    [x0, y0, x1 - x0, y1 - y0]
}

/// Pure-WGPU 3D preview pipeline for points, lines, polylines, and opaque/translucent meshes.
pub struct Preview3dPipeline {
    points_pipeline: wgpu::RenderPipeline,
    lines_pipeline: wgpu::RenderPipeline,
    line_strip_pipeline: wgpu::RenderPipeline,
    opaque_pipeline: wgpu::RenderPipeline,
    translucent_pipeline: wgpu::RenderPipeline,
    translucent_depth_write_pipeline: wgpu::RenderPipeline,
    points_depth_stamp_pipeline: wgpu::RenderPipeline,
    lines_depth_stamp_pipeline: wgpu::RenderPipeline,
    line_strip_depth_stamp_pipeline: wgpu::RenderPipeline,
    triangles_depth_stamp_pipeline: wgpu::RenderPipeline,
    depth_peel: Option<DepthPeelPipelines>,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    vertex_buffer: DynamicGpuBuffer,
    index_buffer: DynamicGpuBuffer,
    staged_vertices: Vec<PreviewVertex3D>,
    staged_indices: Vec<u32>,
    staged_draws: Vec<StagedDraw>,
}

#[allow(clippy::too_many_arguments)]
fn create_subpipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    label: &str,
    vs_entry: &str,
    fs_entry: &str,
    topology: wgpu::PrimitiveTopology,
    color_format: wgpu::TextureFormat,
    depth_format: Option<wgpu::TextureFormat>,
    depth_write: bool,
    depth_compare: wgpu::CompareFunction,
) -> wgpu::RenderPipeline {
    let vertex_layout = preview_vertex_layout();

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some(vs_entry),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[Some(vertex_layout)],
        },
        primitive: wgpu::PrimitiveState {
            topology,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: depth_format.map(|format| wgpu::DepthStencilState {
            format,
            depth_write_enabled: Some(depth_write),
            depth_compare: Some(depth_compare),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some(fs_entry),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: color_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

impl Preview3dPipeline {
    pub fn new(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        depth_format: Option<wgpu::TextureFormat>,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tricore_preview3d_shader"),
            source: wgpu::ShaderSource::Wgsl(PREVIEW3D_WGSL.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tricore_preview3d_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tricore_preview3d_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tricore_preview3d_camera"),
            size: std::mem::size_of::<PreviewCameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tricore_preview3d_camera_bg"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        let points_pipeline = create_subpipeline(
            device,
            &pipeline_layout,
            &shader,
            "tricore_preview3d_points",
            "vs_main",
            "fs_main",
            wgpu::PrimitiveTopology::PointList,
            color_format,
            depth_format,
            true,
            wgpu::CompareFunction::LessEqual,
        );
        let lines_pipeline = create_subpipeline(
            device,
            &pipeline_layout,
            &shader,
            "tricore_preview3d_lines",
            "vs_main",
            "fs_main",
            wgpu::PrimitiveTopology::LineList,
            color_format,
            depth_format,
            true,
            wgpu::CompareFunction::LessEqual,
        );
        let line_strip_pipeline = create_subpipeline(
            device,
            &pipeline_layout,
            &shader,
            "tricore_preview3d_line_strip",
            "vs_main",
            "fs_main",
            wgpu::PrimitiveTopology::LineStrip,
            color_format,
            depth_format,
            true,
            wgpu::CompareFunction::LessEqual,
        );
        let opaque_pipeline = create_subpipeline(
            device,
            &pipeline_layout,
            &shader,
            "tricore_preview3d_opaque",
            "vs_main",
            "fs_main",
            wgpu::PrimitiveTopology::TriangleList,
            color_format,
            depth_format,
            true,
            wgpu::CompareFunction::LessEqual,
        );
        let translucent_pipeline = create_subpipeline(
            device,
            &pipeline_layout,
            &shader,
            "tricore_preview3d_translucent",
            "vs_main",
            "fs_main",
            wgpu::PrimitiveTopology::TriangleList,
            color_format,
            depth_format,
            false,
            wgpu::CompareFunction::LessEqual,
        );
        let translucent_depth_write_pipeline = create_subpipeline(
            device,
            &pipeline_layout,
            &shader,
            "tricore_preview3d_translucent_depth_write",
            "vs_main",
            "fs_main",
            wgpu::PrimitiveTopology::TriangleList,
            color_format,
            depth_format,
            true,
            wgpu::CompareFunction::LessEqual,
        );
        let points_depth_stamp_pipeline = create_subpipeline(
            device,
            &pipeline_layout,
            &shader,
            "tricore_preview3d_points_depth_stamp",
            "vs_depth_stamp",
            "fs_depth_stamp",
            wgpu::PrimitiveTopology::PointList,
            color_format,
            depth_format,
            true,
            wgpu::CompareFunction::Always,
        );
        let lines_depth_stamp_pipeline = create_subpipeline(
            device,
            &pipeline_layout,
            &shader,
            "tricore_preview3d_lines_depth_stamp",
            "vs_depth_stamp",
            "fs_depth_stamp",
            wgpu::PrimitiveTopology::LineList,
            color_format,
            depth_format,
            true,
            wgpu::CompareFunction::Always,
        );
        let line_strip_depth_stamp_pipeline = create_subpipeline(
            device,
            &pipeline_layout,
            &shader,
            "tricore_preview3d_line_strip_depth_stamp",
            "vs_depth_stamp",
            "fs_depth_stamp",
            wgpu::PrimitiveTopology::LineStrip,
            color_format,
            depth_format,
            true,
            wgpu::CompareFunction::Always,
        );
        let triangles_depth_stamp_pipeline = create_subpipeline(
            device,
            &pipeline_layout,
            &shader,
            "tricore_preview3d_triangles_depth_stamp",
            "vs_depth_stamp",
            "fs_depth_stamp",
            wgpu::PrimitiveTopology::TriangleList,
            color_format,
            depth_format,
            true,
            wgpu::CompareFunction::Always,
        );

        let depth_peel = depth_format.map(|df| {
            DepthPeelPipelines::new(device, &bind_group_layout, &shader, color_format, df)
        });

        Self {
            points_pipeline,
            lines_pipeline,
            line_strip_pipeline,
            opaque_pipeline,
            translucent_pipeline,
            translucent_depth_write_pipeline,
            points_depth_stamp_pipeline,
            lines_depth_stamp_pipeline,
            line_strip_depth_stamp_pipeline,
            triangles_depth_stamp_pipeline,
            depth_peel,
            camera_buffer,
            camera_bind_group,
            vertex_buffer: DynamicGpuBuffer::new(
                device,
                4096,
                wgpu::BufferUsages::VERTEX,
                "tricore_preview3d_vertices",
            ),
            index_buffer: DynamicGpuBuffer::new(
                device,
                1024,
                wgpu::BufferUsages::INDEX,
                "tricore_preview3d_indices",
            ),
            staged_vertices: Vec::new(),
            staged_indices: Vec::new(),
            staged_draws: Vec::new(),
        }
    }

    pub fn set_camera(&self, queue: &wgpu::Queue, view_proj: Mat4) {
        let uniform = PreviewCameraUniform {
            view_proj: view_proj.to_cols_array_2d(),
        };
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&uniform));
    }

    pub fn set_raw_camera(&self, queue: &wgpu::Queue, camera: &RawCamera) {
        self.set_camera(queue, camera.view_proj_matrix());
    }

    fn stage_slice(
        &mut self,
        kind: PreviewPipelineKind,
        verts: &[PreviewVertex3D],
        indices: Option<&[u32]>,
    ) {
        if verts.is_empty() {
            return;
        }
        let vertex_start = self.staged_vertices.len() as u32;
        self.staged_vertices.extend_from_slice(verts);
        let vertex_count = verts.len() as u32;

        if let Some(idx) = indices {
            if idx.is_empty() {
                return;
            }
            let index_start = self.staged_indices.len() as u32;
            self.staged_indices.extend_from_slice(idx);
            self.staged_draws.push(StagedDraw {
                kind,
                vertex_start,
                vertex_count,
                index_start,
                index_count: idx.len() as u32,
                indexed: true,
            });
        } else {
            self.staged_draws.push(StagedDraw {
                kind,
                vertex_start,
                vertex_count,
                index_start: 0,
                index_count: 0,
                indexed: false,
            });
        }
    }

    /// Uploads camera and all primitive batches for a frame. Call [`Self::draw_prepared`] inside a pass.
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view_proj: Mat4,
        data: &Preview3dPassData<'_>,
    ) {
        self.set_camera(queue, view_proj);
        self.staged_vertices.clear();
        self.staged_indices.clear();
        self.staged_draws.clear();

        self.stage_slice(
            PreviewPipelineKind::OpaqueTriangles,
            data.opaque_triangles,
            data.opaque_indices,
        );
        self.stage_slice(PreviewPipelineKind::Lines, data.lines, None);
        for strip in data.line_strips {
            if strip.len() >= 2 {
                self.stage_slice(PreviewPipelineKind::LineStrip, strip, None);
            }
        }
        self.stage_slice(PreviewPipelineKind::Points, data.points, None);
        self.stage_slice(
            PreviewPipelineKind::TranslucentTriangles,
            data.translucent_triangles,
            data.translucent_indices,
        );

        self.vertex_buffer
            .write_pod(device, queue, &self.staged_vertices);
        self.index_buffer
            .write_pod(device, queue, &self.staged_indices);
    }

    /// Records draw calls for the last [`Self::prepare`] call into an active render pass.
    pub fn draw_prepared<'rp>(&'rp self, pass: &mut wgpu::RenderPass<'rp>) {
        self.draw_prepared_with_mode(pass, Preview3dDepthMode::Standard);
    }

    /// Records draw calls for the last [`Self::prepare`] call into an active render pass using `mode`.
    pub fn draw_prepared_with_mode<'rp>(
        &'rp self,
        pass: &mut wgpu::RenderPass<'rp>,
        mode: Preview3dDepthMode,
    ) {
        if self.staged_draws.is_empty() {
            return;
        }
        pass.set_bind_group(0, &self.camera_bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.buffer.slice(..));
        if !self.staged_indices.is_empty() {
            pass.set_index_buffer(self.index_buffer.buffer.slice(..), wgpu::IndexFormat::Uint32);
        }

        for draw in &self.staged_draws {
            let pipeline = match mode {
                Preview3dDepthMode::Standard => match draw.kind {
                    PreviewPipelineKind::Points => &self.points_pipeline,
                    PreviewPipelineKind::Lines => &self.lines_pipeline,
                    PreviewPipelineKind::LineStrip => &self.line_strip_pipeline,
                    PreviewPipelineKind::OpaqueTriangles => &self.opaque_pipeline,
                    PreviewPipelineKind::TranslucentTriangles => &self.translucent_pipeline,
                },
                Preview3dDepthMode::DepthTestedTranslucentWrite => match draw.kind {
                    PreviewPipelineKind::Points => &self.points_pipeline,
                    PreviewPipelineKind::Lines => &self.lines_pipeline,
                    PreviewPipelineKind::LineStrip => &self.line_strip_pipeline,
                    PreviewPipelineKind::OpaqueTriangles => &self.opaque_pipeline,
                    PreviewPipelineKind::TranslucentTriangles => {
                        &self.translucent_depth_write_pipeline
                    }
                },
                Preview3dDepthMode::DepthStamp => match draw.kind {
                    PreviewPipelineKind::Points => &self.points_depth_stamp_pipeline,
                    PreviewPipelineKind::Lines => &self.lines_depth_stamp_pipeline,
                    PreviewPipelineKind::LineStrip => &self.line_strip_depth_stamp_pipeline,
                    PreviewPipelineKind::OpaqueTriangles
                    | PreviewPipelineKind::TranslucentTriangles => {
                        &self.triangles_depth_stamp_pipeline
                    }
                },
            };
            pass.set_pipeline(pipeline);
            if draw.indexed {
                pass.draw_indexed(
                    draw.index_start..draw.index_start + draw.index_count,
                    draw.vertex_start as i32,
                    0..1,
                );
            } else {
                pass.draw(draw.vertex_start..draw.vertex_start + draw.vertex_count, 0..1);
            }
        }
    }

    /// Prepares geometry and records a complete 3D preview render pass into `encoder`.
    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        depth_view: Option<&wgpu::TextureView>,
        clear_color: Option<wgpu::Color>,
        clear_depth: Option<f32>,
        view_proj: Mat4,
        data: &Preview3dPassData<'_>,
    ) {
        self.render_with_mode(
            device,
            queue,
            encoder,
            color_view,
            depth_view,
            clear_color,
            clear_depth,
            view_proj,
            data,
            Preview3dDepthMode::Standard,
        );
    }

    /// Returns `true` if lazy VRAM scratch buffers for depth peeling are currently allocated.
    pub fn has_depth_peel_scratch(&self) -> bool {
        self.depth_peel
            .as_ref()
            .is_some_and(|peel| peel.scratch.is_some())
    }

    /// Drops any lazily-allocated depth-peel scratch buffers (e.g. on viewport resize).
    pub fn clear_depth_peel_scratch(&mut self) {
        if let Some(ref mut peel) = self.depth_peel {
            peel.scratch = None;
        }
    }

    /// Prepares geometry and records a depth-tested 3D pass.
    ///
    /// When `data.has_translucent()` is `false`, executes the zero-translucent fast-path directly
    /// into `color_view` / `depth_view` without allocating depth-peeling scratch textures.
    /// When translucent triangles are present and `depth_view` is `Some`, executes the D-P64-10
    /// 6-subpass front-to-back depth peeling (`K = 4`) + `Rgba16Float` 2-MRT tail + static 64x64
    /// Void-and-Cluster blue-noise importance-sampled single-depth resolve into `color_view` / `depth_view`.
    #[allow(clippy::too_many_arguments)]
    pub fn render_depth_tested(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        depth_view: Option<&wgpu::TextureView>,
        target_size: [u32; 2],
        clear_color: Option<wgpu::Color>,
        clear_depth: Option<f32>,
        view_proj: Mat4,
        data: &Preview3dPassData<'_>,
    ) {
        if !data.has_translucent() || depth_view.is_none() || self.depth_peel.is_none() {
            self.render_with_mode(
                device,
                queue,
                encoder,
                color_view,
                depth_view,
                clear_color,
                clear_depth,
                view_proj,
                data,
                Preview3dDepthMode::DepthTestedTranslucentWrite,
            );
            return;
        }

        let main_depth_view = depth_view.expect("checked Some above");
        let phys_w = target_size[0].max(1);
        let phys_h = target_size[1].max(1);

        self.prepare(device, queue, view_proj, data);
        let scissor =
            compute_translucent_scissor_rect(data.translucent_triangles, view_proj, phys_w, phys_h);
        let has_scissor_area = scissor[2] > 0 && scissor[3] > 0;

        let peel = self.depth_peel.as_mut().expect("checked Some above");
        peel.ensure_blue_noise_uploaded(queue);
        let _ = peel.ensure_scratch(device, phys_w, phys_h);

        let peel = self.depth_peel.as_ref().expect("checked Some above");
        let scratch = peel.scratch.as_ref().expect("scratch allocated");

        let has_indices = !self.staged_indices.is_empty();

        // Sub-pass 0: Opaque triangles + lines + line strips + points -> opaque_target.
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("tricore_preview3d_peel_opaque"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &scratch.opaque_color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear_color.unwrap_or(wgpu::Color::TRANSPARENT)),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &scratch.opaque_depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear_depth.unwrap_or(1.0)),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, &self.camera_bind_group, &[]);
            pass.set_vertex_buffer(0, self.vertex_buffer.buffer.slice(..));
            if has_indices {
                pass.set_index_buffer(self.index_buffer.buffer.slice(..), wgpu::IndexFormat::Uint32);
            }
            for draw in &self.staged_draws {
                let pipeline = match draw.kind {
                    PreviewPipelineKind::Points => &self.points_pipeline,
                    PreviewPipelineKind::Lines => &self.lines_pipeline,
                    PreviewPipelineKind::LineStrip => &self.line_strip_pipeline,
                    PreviewPipelineKind::OpaqueTriangles => &self.opaque_pipeline,
                    PreviewPipelineKind::TranslucentTriangles => continue,
                };
                pass.set_pipeline(pipeline);
                if draw.indexed {
                    pass.draw_indexed(
                        draw.index_start..draw.index_start + draw.index_count,
                        draw.vertex_start as i32,
                        0..1,
                    );
                } else {
                    pass.draw(draw.vertex_start..draw.vertex_start + draw.vertex_count, 0..1);
                }
            }
        }

        // Sub-passes 1..4: Front-to-back depth peeling passes p = 0..4 (single shared RenderPipeline).
        for p in 0..PEEL_LAYER_COUNT {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("tricore_preview3d_peel_layer"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &scratch.peel_color_views[p],
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &scratch.peel_depth_views[p],
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if has_scissor_area {
                pass.set_scissor_rect(scissor[0], scissor[1], scissor[2], scissor[3]);
                pass.set_pipeline(&peel.peel_pipeline);
                pass.set_bind_group(0, &self.camera_bind_group, &[]);
                pass.set_bind_group(1, &scratch.peel_bind_groups[p], &[]);
                pass.set_vertex_buffer(0, self.vertex_buffer.buffer.slice(..));
                if has_indices {
                    pass.set_index_buffer(
                        self.index_buffer.buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                }
                for draw in &self.staged_draws {
                    if !matches!(draw.kind, PreviewPipelineKind::TranslucentTriangles) {
                        continue;
                    }
                    if draw.indexed {
                        pass.draw_indexed(
                            draw.index_start..draw.index_start + draw.index_count,
                            draw.vertex_start as i32,
                            0..1,
                        );
                    } else {
                        pass.draw(draw.vertex_start..draw.vertex_start + draw.vertex_count, 0..1);
                    }
                }
            }
        }

        // Sub-pass 5: Tail pass (k > 4) into 2 Rgba16Float MRTs (no depth attachment).
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("tricore_preview3d_peel_tail"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &scratch.tail_accum_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.0,
                                g: 0.0,
                                b: 0.0,
                                a: 0.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &scratch.tail_meta_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 1.0,
                                g: 1.0,
                                b: 1.0,
                                a: 0.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    }),
                ],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if has_scissor_area {
                pass.set_scissor_rect(scissor[0], scissor[1], scissor[2], scissor[3]);
                pass.set_pipeline(&peel.tail_pipeline);
                pass.set_bind_group(0, &self.camera_bind_group, &[]);
                pass.set_bind_group(1, &scratch.tail_bind_group, &[]);
                pass.set_vertex_buffer(0, self.vertex_buffer.buffer.slice(..));
                if has_indices {
                    pass.set_index_buffer(
                        self.index_buffer.buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                }
                for draw in &self.staged_draws {
                    if !matches!(draw.kind, PreviewPipelineKind::TranslucentTriangles) {
                        continue;
                    }
                    if draw.indexed {
                        pass.draw_indexed(
                            draw.index_start..draw.index_start + draw.index_count,
                            draw.vertex_start as i32,
                            0..1,
                        );
                    } else {
                        pass.draw(draw.vertex_start..draw.vertex_start + draw.vertex_count, 0..1);
                    }
                }
            }
        }

        // Sub-pass 6: Fullscreen Resolve -> main_target.color + main_target.depth.
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("tricore_preview3d_peel_resolve"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: main_depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&peel.resolve_pipeline);
            pass.set_bind_group(0, &scratch.resolve_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
    }

    /// Prepares geometry and records an always-on-top 3D pass that unconditionally stamps `z = 0.0`.
    #[allow(clippy::too_many_arguments)]
    pub fn render_depth_stamp(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        depth_view: Option<&wgpu::TextureView>,
        clear_color: Option<wgpu::Color>,
        clear_depth: Option<f32>,
        view_proj: Mat4,
        data: &Preview3dPassData<'_>,
    ) {
        self.render_with_mode(
            device,
            queue,
            encoder,
            color_view,
            depth_view,
            clear_color,
            clear_depth,
            view_proj,
            data,
            Preview3dDepthMode::DepthStamp,
        );
    }

    /// Prepares geometry and records a complete 3D preview render pass into `encoder` using `mode`.
    #[allow(clippy::too_many_arguments)]
    pub fn render_with_mode(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        depth_view: Option<&wgpu::TextureView>,
        clear_color: Option<wgpu::Color>,
        clear_depth: Option<f32>,
        view_proj: Mat4,
        data: &Preview3dPassData<'_>,
        mode: Preview3dDepthMode,
    ) {
        if data.is_empty() && clear_color.is_none() && clear_depth.is_none() {
            return;
        }
        self.prepare(device, queue, view_proj, data);

        let color_load = match clear_color {
            Some(c) => wgpu::LoadOp::Clear(c),
            None => wgpu::LoadOp::Load,
        };
        let depth_attachment = depth_view.map(|view| wgpu::RenderPassDepthStencilAttachment {
            view,
            depth_ops: Some(wgpu::Operations {
                load: match clear_depth {
                    Some(d) => wgpu::LoadOp::Clear(d),
                    None => wgpu::LoadOp::Load,
                },
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        });

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("tricore_preview3d_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: color_load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: depth_attachment,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        self.draw_prepared_with_mode(&mut pass, mode);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::readback::{DepthReadbackOptions, DepthReadbackSurface};
    use crate::renderer::{GpuContext, GpuInitError};
    use crate::targets::{OffscreenTarget, TargetDimensions};

    #[test]
    fn preview3d_translucent_depth_write_and_depth_stamp_override() {
        let gpu = match pollster::block_on(GpuContext::new_headless_for_platform(
            wgpu::PowerPreference::HighPerformance,
        )) {
            Ok(g) => g,
            Err(GpuInitError::NoAdapter(_)) => return,
            Err(e) => panic!("GPU init failed: {e}"),
        };

        let w = 32_u32;
        let h = 32_u32;
        let target = OffscreenTarget::new(
            &gpu.device,
            TargetDimensions::new(w, h, 1.0),
            wgpu::TextureFormat::Rgba8Unorm,
            Some(wgpu::TextureFormat::Depth32Float),
        );
        let mut dt_pipeline = Preview3dPipeline::new(
            &gpu.device,
            wgpu::TextureFormat::Rgba8Unorm,
            Some(wgpu::TextureFormat::Depth32Float),
        );
        let mut aot_pipeline = Preview3dPipeline::new(
            &gpu.device,
            wgpu::TextureFormat::Rgba8Unorm,
            Some(wgpu::TextureFormat::Depth32Float),
        );
        let mut depth_rb = DepthReadbackSurface::new(DepthReadbackOptions { y_flip: false });
        depth_rb.ensure_dimensions(&gpu.device, w, h);

        // Identity view_proj: clip coords == vertex positions directly (WGPU NDC z in [0, 1]).
        // Translucent quad at z = 0.5 covering NDC [-0.8, 0.8]^2 with alpha = 0.5.
        let translucent_quad = [
            PreviewVertex3D::new([-0.8, -0.8, 0.5], [0.0, 1.0, 0.0, 0.5]),
            PreviewVertex3D::new([0.8, -0.8, 0.5], [0.0, 1.0, 0.0, 0.5]),
            PreviewVertex3D::new([0.8, 0.8, 0.5], [0.0, 1.0, 0.0, 0.5]),
            PreviewVertex3D::new([-0.8, 0.8, 0.5], [0.0, 1.0, 0.0, 0.5]),
        ];
        let quad_indices = [0_u32, 1, 2, 0, 2, 3];

        // AlwaysOnTop quad at z = 0.9 (farther!) covering right half NDC [0.0, 0.8] x [-0.8, 0.8].
        let aot_quad = [
            PreviewVertex3D::new([0.0, -0.8, 0.9], [1.0, 0.0, 0.0, 1.0]),
            PreviewVertex3D::new([0.8, -0.8, 0.9], [1.0, 0.0, 0.0, 1.0]),
            PreviewVertex3D::new([0.8, 0.8, 0.9], [1.0, 0.0, 0.0, 1.0]),
            PreviewVertex3D::new([0.0, 0.8, 0.9], [1.0, 0.0, 0.0, 1.0]),
        ];

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("preview3d_depth_modes_test"),
            });

        dt_pipeline.render_depth_tested(
            &gpu.device,
            &gpu.queue,
            &mut encoder,
            &target.color_view,
            target.depth_view.as_ref(),
            [w, h],
            Some(wgpu::Color::TRANSPARENT),
            Some(1.0),
            Mat4::IDENTITY,
            &Preview3dPassData {
                translucent_triangles: &translucent_quad,
                translucent_indices: Some(&quad_indices),
                ..Default::default()
            },
        );

        aot_pipeline.render_depth_stamp(
            &gpu.device,
            &gpu.queue,
            &mut encoder,
            &target.color_view,
            target.depth_view.as_ref(),
            None,
            None,
            Mat4::IDENTITY,
            &Preview3dPassData {
                opaque_triangles: &aot_quad,
                opaque_indices: Some(&quad_indices),
                ..Default::default()
            },
        );

        depth_rb
            .copy_from_texture(&mut encoder, target.depth_texture.as_ref().unwrap())
            .unwrap();
        gpu.queue.submit(std::iter::once(encoder.finish()));

        let depths = depth_rb.map_and_repack(&gpu.device).unwrap();
        // Left half (x=8, y=16): non-overlapped translucent DepthTested quad must have z ≈ 0.5.
        let dt_z = depths[(16 * w + 8) as usize];
        assert!(
            (dt_z - 0.5).abs() < 1e-3,
            "expected translucent DepthTested z ≈ 0.5, got {dt_z}"
        );
        // Right half (x=24, y=16): overlapped by AlwaysOnTop depth_stamp quad -> z == 0.0.
        let aot_z = depths[(16 * w + 24) as usize];
        assert_eq!(aot_z, 0.0, "expected AlwaysOnTop depth_stamp z == 0.0, got {aot_z}");
        // Corner (x=0, y=0): uncovered background -> z == 1.0.
        let bg_z = depths[0];
        assert_eq!(bg_z, 1.0, "expected background z == 1.0, got {bg_z}");
    }
}
