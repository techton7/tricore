//! Generic transparent texture alpha-blit composite pass (`LoadOp::Load`, zero Vello dependency).

const ALPHA_BLIT_WGSL: &str = r#"
struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    let uv = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    let pos = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    return VsOut(pos, uv);
}

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_samp: sampler;

@fragment
fn fs_main_straight(in: VsOut) -> @location(0) vec4<f32> {
    let c = textureSample(src_tex, src_samp, in.uv);
    return vec4<f32>(c.rgb * c.a, c.a);
}

@fragment
fn fs_main_premul(in: VsOut) -> @location(0) vec4<f32> {
    return textureSample(src_tex, src_samp, in.uv);
}
"#;

/// Alpha encoding of the source texture sampled by [`AlphaBlitPipeline`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SourceAlphaMode {
    /// Source texture stores straight (unpremultiplied) RGB + A (e.g., Vello `Rgba8Unorm` output).
    #[default]
    Straight,
    /// Source texture already stores premultiplied RGB + A.
    Premultiplied,
}

/// Pure-WGPU fullscreen alpha-compositing blit pipeline using `LoadOp::Load`.
pub struct AlphaBlitPipeline {
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    straight_pipeline: wgpu::RenderPipeline,
    premul_pipeline: wgpu::RenderPipeline,
}

fn create_blit_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    dst_format: wgpu::TextureFormat,
    fs_entry: &str,
    label: &str,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some(fs_entry),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: dst_format,
                blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

impl AlphaBlitPipeline {
    pub fn new(device: &wgpu::Device, dst_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tricore_alpha_blit_shader"),
            source: wgpu::ShaderSource::Wgsl(ALPHA_BLIT_WGSL.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tricore_alpha_blit_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("tricore_alpha_blit_sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tricore_alpha_blit_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let straight_pipeline = create_blit_pipeline(
            device,
            &pipeline_layout,
            &shader,
            dst_format,
            "fs_main_straight",
            "tricore_alpha_blit_straight",
        );
        let premul_pipeline = create_blit_pipeline(
            device,
            &pipeline_layout,
            &shader,
            dst_format,
            "fs_main_premul",
            "tricore_alpha_blit_premul",
        );

        Self {
            bind_group_layout,
            sampler,
            straight_pipeline,
            premul_pipeline,
        }
    }

    /// Creates a reusable bind group for a given source texture view.
    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        src_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tricore_alpha_blit_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(src_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        })
    }

    /// Composites a pre-bound source texture over `dst_view` preserving existing contents (`LoadOp::Load`).
    pub fn composite_with_bind_group(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        dst_view: &wgpu::TextureView,
        mode: SourceAlphaMode,
    ) {
        let pipeline = match mode {
            SourceAlphaMode::Straight => &self.straight_pipeline,
            SourceAlphaMode::Premultiplied => &self.premul_pipeline,
        };

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("tricore_alpha_blit_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: dst_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Convenience method creating a temporary bind group and compositing `src_view` over `dst_view`.
    pub fn composite(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        src_view: &wgpu::TextureView,
        dst_view: &wgpu::TextureView,
        mode: SourceAlphaMode,
    ) {
        let bg = self.create_bind_group(device, src_view);
        self.composite_with_bind_group(encoder, &bg, dst_view, mode);
    }
}
