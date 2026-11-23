//! Depth-tested 3D tool preview pipeline (points, lines, line strips, opaque & translucent triangles).

use glam::Mat4;

use crate::camera::RawCamera;
use crate::pipelines::DynamicGpuBuffer;

const PREVIEW3D_WGSL: &str = r#"
struct CameraUniform {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) color: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip_pos = camera.view_proj * vec4<f32>(in.pos, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return in.color;
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

/// Pure-WGPU 3D preview pipeline for points, lines, polylines, and opaque/translucent meshes.
pub struct Preview3dPipeline {
    points_pipeline: wgpu::RenderPipeline,
    lines_pipeline: wgpu::RenderPipeline,
    line_strip_pipeline: wgpu::RenderPipeline,
    opaque_pipeline: wgpu::RenderPipeline,
    translucent_pipeline: wgpu::RenderPipeline,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    vertex_buffer: DynamicGpuBuffer,
    index_buffer: DynamicGpuBuffer,
    staged_vertices: Vec<PreviewVertex3D>,
    staged_indices: Vec<u32>,
    staged_draws: Vec<StagedDraw>,
}

fn create_subpipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    label: &str,
    topology: wgpu::PrimitiveTopology,
    color_format: wgpu::TextureFormat,
    depth_format: Option<wgpu::TextureFormat>,
    depth_write: bool,
) -> wgpu::RenderPipeline {
    let vertex_layout = wgpu::VertexBufferLayout {
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
    };

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
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
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
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
            wgpu::PrimitiveTopology::PointList,
            color_format,
            depth_format,
            true,
        );
        let lines_pipeline = create_subpipeline(
            device,
            &pipeline_layout,
            &shader,
            "tricore_preview3d_lines",
            wgpu::PrimitiveTopology::LineList,
            color_format,
            depth_format,
            true,
        );
        let line_strip_pipeline = create_subpipeline(
            device,
            &pipeline_layout,
            &shader,
            "tricore_preview3d_line_strip",
            wgpu::PrimitiveTopology::LineStrip,
            color_format,
            depth_format,
            true,
        );
        let opaque_pipeline = create_subpipeline(
            device,
            &pipeline_layout,
            &shader,
            "tricore_preview3d_opaque",
            wgpu::PrimitiveTopology::TriangleList,
            color_format,
            depth_format,
            true,
        );
        let translucent_pipeline = create_subpipeline(
            device,
            &pipeline_layout,
            &shader,
            "tricore_preview3d_translucent",
            wgpu::PrimitiveTopology::TriangleList,
            color_format,
            depth_format,
            false,
        );

        Self {
            points_pipeline,
            lines_pipeline,
            line_strip_pipeline,
            opaque_pipeline,
            translucent_pipeline,
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
        if self.staged_draws.is_empty() {
            return;
        }
        pass.set_bind_group(0, &self.camera_bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.buffer.slice(..));
        if !self.staged_indices.is_empty() {
            pass.set_index_buffer(self.index_buffer.buffer.slice(..), wgpu::IndexFormat::Uint32);
        }

        for draw in &self.staged_draws {
            let pipeline = match draw.kind {
                PreviewPipelineKind::Points => &self.points_pipeline,
                PreviewPipelineKind::Lines => &self.lines_pipeline,
                PreviewPipelineKind::LineStrip => &self.line_strip_pipeline,
                PreviewPipelineKind::OpaqueTriangles => &self.opaque_pipeline,
                PreviewPipelineKind::TranslucentTriangles => &self.translucent_pipeline,
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

        self.draw_prepared(&mut pass);
    }
}
