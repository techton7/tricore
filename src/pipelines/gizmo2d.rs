//! 2D screen-space triangle pipeline for gizmos and pixel-space vector overlays.

use crate::math::ViewportRect;
use crate::pipelines::DynamicGpuBuffer;

const GIZMO2D_WGSL: &str = r#"
struct ViewportUniform {
    origin: vec2<f32>,
    size: vec2<f32>,
    encode_srgb: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0)
var<uniform> vp: ViewportUniform;

struct VsIn {
    @location(0) pos_px: vec2<f32>,
    @location(1) color: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let norm = (in.pos_px - vp.origin) / max(vp.size, vec2<f32>(1.0, 1.0));
    let ndc = vec2<f32>(norm.x * 2.0 - 1.0, 1.0 - norm.y * 2.0);
    var out: VsOut;
    out.clip_pos = vec4<f32>(ndc, 0.0, 1.0);
    out.color = in.color;
    return out;
}

fn linear_to_srgb_ch(c: f32) -> f32 {
    let cl = clamp(c, 0.0, 1.0);
    return select(1.055 * pow(cl, 1.0 / 2.4) - 0.055, cl * 12.92, cl <= 0.0031308);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let a = clamp(in.color.a, 0.0, 1.0);
    if (a <= 1e-6) {
        discard;
    }
    if (vp.encode_srgb == 0u) {
        return in.color;
    }
    let straight = in.color.rgb / a;
    let srgb = vec3<f32>(
        linear_to_srgb_ch(straight.r),
        linear_to_srgb_ch(straight.g),
        linear_to_srgb_ch(straight.b),
    );
    return vec4<f32>(srgb * a, a);
}
"#;

/// Struct-of-Arrays screen-space triangle batch in top-left pixel coordinates and linear premultiplied RGBA.
#[derive(Clone, Copy, Debug, Default)]
pub struct ScreenTriangleBatch<'a> {
    pub vertices: &'a [[f32; 2]],
    pub colors: &'a [[f32; 4]],
    pub indices: &'a [u32],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct GizmoViewportUniform {
    origin: [f32; 2],
    size: [f32; 2],
    encode_srgb: u32,
    _pad: [u32; 3],
}

/// Pure-WGPU 2D screen-triangle pipeline (`PREMULTIPLIED_ALPHA_BLENDING`, depth compare `Always`).
pub struct ScreenTrianglePipeline {
    color_only_pipeline: wgpu::RenderPipeline,
    depth_pipeline: Option<wgpu::RenderPipeline>,
    depth_stamp_pipeline: Option<wgpu::RenderPipeline>,
    encode_srgb: u32,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    vertex_buffer: DynamicGpuBuffer,
    color_buffer: DynamicGpuBuffer,
    index_buffer: DynamicGpuBuffer,
    prepared_index_count: u32,
}

fn create_gizmo_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    color_format: wgpu::TextureFormat,
    depth_format: Option<wgpu::TextureFormat>,
    depth_write: bool,
) -> wgpu::RenderPipeline {
    let pos_layout = wgpu::VertexBufferLayout {
        array_stride: 8,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x2,
            offset: 0,
            shader_location: 0,
        }],
    };
    let color_layout = wgpu::VertexBufferLayout {
        array_stride: 16,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x4,
            offset: 0,
            shader_location: 1,
        }],
    };

    let label = match (depth_format.is_some(), depth_write) {
        (false, _) => "tricore_gizmo2d_color_pipeline",
        (true, false) => "tricore_gizmo2d_depth_pipeline",
        (true, true) => "tricore_gizmo2d_depth_stamp_pipeline",
    };

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[Some(pos_layout), Some(color_layout)],
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
        depth_stencil: depth_format.map(|format| wgpu::DepthStencilState {
            format,
            depth_write_enabled: Some(depth_write),
            depth_compare: Some(wgpu::CompareFunction::Always),
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
                blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

impl ScreenTrianglePipeline {
    pub fn new(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        depth_format: Option<wgpu::TextureFormat>,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tricore_gizmo2d_shader"),
            source: wgpu::ShaderSource::Wgsl(GIZMO2D_WGSL.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tricore_gizmo2d_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tricore_gizmo2d_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tricore_gizmo2d_uniform"),
            size: std::mem::size_of::<GizmoViewportUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tricore_gizmo2d_bg"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let color_only_pipeline =
            create_gizmo_pipeline(device, &pipeline_layout, &shader, color_format, None, false);
        let depth_pipeline = depth_format.map(|df| {
            create_gizmo_pipeline(device, &pipeline_layout, &shader, color_format, Some(df), false)
        });
        let depth_stamp_pipeline = depth_format.map(|df| {
            create_gizmo_pipeline(device, &pipeline_layout, &shader, color_format, Some(df), true)
        });

        Self {
            color_only_pipeline,
            depth_pipeline,
            depth_stamp_pipeline,
            encode_srgb: u32::from(!color_format.is_srgb()),
            uniform_buffer,
            bind_group,
            vertex_buffer: DynamicGpuBuffer::new(
                device,
                2048,
                wgpu::BufferUsages::VERTEX,
                "tricore_gizmo2d_vertices",
            ),
            color_buffer: DynamicGpuBuffer::new(
                device,
                4096,
                wgpu::BufferUsages::VERTEX,
                "tricore_gizmo2d_colors",
            ),
            index_buffer: DynamicGpuBuffer::new(
                device,
                1024,
                wgpu::BufferUsages::INDEX,
                "tricore_gizmo2d_indices",
            ),
            prepared_index_count: 0,
        }
    }

    /// Uploads viewport uniform and SoA triangle buffers. Returns `false` if there is nothing to draw.
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        viewport: ViewportRect,
        batch: &ScreenTriangleBatch<'_>,
    ) -> bool {
        if batch.vertices.is_empty()
            || batch.colors.is_empty()
            || batch.indices.is_empty()
            || batch.vertices.len() != batch.colors.len()
        {
            self.prepared_index_count = 0;
            return false;
        }

        let uniform = GizmoViewportUniform {
            origin: [viewport.x, viewport.y],
            size: [viewport.width.max(1.0), viewport.height.max(1.0)],
            encode_srgb: self.encode_srgb,
            _pad: [0; 3],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniform));
        self.vertex_buffer.write_pod(device, queue, batch.vertices);
        self.color_buffer.write_pod(device, queue, batch.colors);
        self.index_buffer.write_pod(device, queue, batch.indices);
        self.prepared_index_count = batch.indices.len() as u32;
        true
    }

    /// Records the prepared screen-triangle draw into an active render pass.
    pub fn draw_prepared<'rp>(&'rp self, pass: &mut wgpu::RenderPass<'rp>, with_depth: bool) {
        self.draw_prepared_inner(pass, with_depth, false);
    }

    /// Records the prepared screen-triangle draw into an active render pass, stamping `z = 0.0` into depth.
    pub fn draw_prepared_with_depth_stamp<'rp>(&'rp self, pass: &mut wgpu::RenderPass<'rp>) {
        self.draw_prepared_inner(pass, true, true);
    }

    fn draw_prepared_inner<'rp>(
        &'rp self,
        pass: &mut wgpu::RenderPass<'rp>,
        with_depth: bool,
        stamp_depth: bool,
    ) {
        if self.prepared_index_count == 0 {
            return;
        }
        let pipeline = match (with_depth, stamp_depth) {
            (false, _) => &self.color_only_pipeline,
            (true, false) => self
                .depth_pipeline
                .as_ref()
                .expect("ScreenTrianglePipeline created without depth_format"),
            (true, true) => self
                .depth_stamp_pipeline
                .as_ref()
                .expect("ScreenTrianglePipeline created without depth_format"),
        };
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.buffer.slice(..));
        pass.set_vertex_buffer(1, self.color_buffer.buffer.slice(..));
        pass.set_index_buffer(self.index_buffer.buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..self.prepared_index_count, 0, 0..1);
    }

    fn render_internal(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        depth_view: Option<&wgpu::TextureView>,
        viewport: ViewportRect,
        batch: &ScreenTriangleBatch<'_>,
        stamp_depth: bool,
    ) {
        if !self.prepare(device, queue, viewport, batch) {
            return;
        }

        let has_depth_pipeline = if stamp_depth {
            self.depth_stamp_pipeline.is_some()
        } else {
            self.depth_pipeline.is_some()
        };
        let attach_depth = depth_view.filter(|_| has_depth_pipeline);
        let depth_attachment = attach_depth.map(|view| wgpu::RenderPassDepthStencilAttachment {
            view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        });

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("tricore_gizmo2d_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: depth_attachment,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        self.draw_prepared_inner(&mut pass, attach_depth.is_some(), stamp_depth);
    }

    /// Prepares the batch and composites it onto `color_view` using `LoadOp::Load`.
    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        depth_view: Option<&wgpu::TextureView>,
        viewport: ViewportRect,
        batch: &ScreenTriangleBatch<'_>,
    ) {
        self.render_internal(
            device, queue, encoder, color_view, depth_view, viewport, batch, false,
        );
    }

    /// Prepares the batch, composites it onto `color_view` (`LoadOp::Load`), and stamps `z = 0.0` into `depth_view`.
    #[allow(clippy::too_many_arguments)]
    pub fn render_with_depth_stamp(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        depth_view: Option<&wgpu::TextureView>,
        viewport: ViewportRect,
        batch: &ScreenTriangleBatch<'_>,
    ) {
        self.render_internal(
            device, queue, encoder, color_view, depth_view, viewport, batch, true,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::readback::{DepthReadbackOptions, DepthReadbackSurface};
    use crate::renderer::{GpuContext, GpuInitError};
    use crate::targets::{OffscreenTarget, TargetDimensions};

    #[test]
    fn gizmo2d_depth_stamp_discards_zero_alpha_fragments() {
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
        let mut gizmo = ScreenTrianglePipeline::new(
            &gpu.device,
            wgpu::TextureFormat::Rgba8Unorm,
            Some(wgpu::TextureFormat::Depth32Float),
        );
        let mut depth_rb = DepthReadbackSurface::new(DepthReadbackOptions { y_flip: false });
        depth_rb.ensure_dimensions(&gpu.device, w, h);

        // Two quads: left half alpha=0.0 (must discard and keep depth=0.6), right half alpha=1.0 (must stamp 0.0).
        let vertices = [
            [0.0, 0.0],
            [16.0, 0.0],
            [16.0, 32.0],
            [0.0, 32.0],
            [16.0, 0.0],
            [32.0, 0.0],
            [32.0, 32.0],
            [16.0, 32.0],
        ];
        let colors = [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ];
        let indices = [0_u32, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7];

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gizmo2d_zero_alpha_discard_test"),
            });
        // Clear depth to 0.6 first.
        {
            let _clear_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("clear_depth_06"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target.color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: target.depth_view.as_ref().unwrap(),
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.6),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }

        gizmo.render_with_depth_stamp(
            &gpu.device,
            &gpu.queue,
            &mut encoder,
            &target.color_view,
            target.depth_view.as_ref(),
            ViewportRect::new(0.0, 0.0, w as f32, h as f32),
            &ScreenTriangleBatch {
                vertices: &vertices,
                colors: &colors,
                indices: &indices,
            },
        );
        depth_rb
            .copy_from_texture(&mut encoder, target.depth_texture.as_ref().unwrap())
            .unwrap();
        gpu.queue.submit(std::iter::once(encoder.finish()));

        let depths = depth_rb.map_and_repack(&gpu.device).unwrap();
        let transparent_z = depths[(16 * w + 8) as usize];
        assert!(
            (transparent_z - 0.6).abs() < 1e-4,
            "zero-alpha gizmo fragment must discard and keep depth=0.6, got {transparent_z}"
        );
        let opaque_z = depths[(16 * w + 24) as usize];
        assert_eq!(
            opaque_z, 0.0,
            "opaque gizmo fragment must stamp depth=0.0, got {opaque_z}"
        );
    }
}
