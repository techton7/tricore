//! Single-pass fullscreen composite + alpha-finalize and depth-stamp pipelines (zero Vello dependency).

use std::fmt;

use crate::pipelines::SourceAlphaMode;

const COMPOSITE_WGSL: &str = r#"
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

@group(0) @binding(0) var base_tex: texture_2d<f32>;
@group(0) @binding(1) var overlay_tex: texture_2d<f32>;
@group(0) @binding(2) var tex_samp: sampler;

fn to_premul(c: vec4<f32>, straight: bool) -> vec4<f32> {
    if (straight) {
        return vec4<f32>(c.rgb * c.a, c.a);
    }
    return c;
}

fn finalize_pm(pm: vec4<f32>, out_straight: bool) -> vec4<f32> {
    if (out_straight) {
        let rgb = select(vec3<f32>(0.0), clamp(pm.rgb / pm.a, vec3<f32>(0.0), vec3<f32>(1.0)), pm.a > 1e-5);
        return vec4<f32>(rgb, clamp(pm.a, 0.0, 1.0));
    }
    return pm;
}

fn composite_impl(uv: vec2<f32>, base_straight: bool, overlay_straight: bool, out_straight: bool) -> vec4<f32> {
    let base_pm = to_premul(textureSample(base_tex, tex_samp, uv), base_straight);
    let over_pm = to_premul(textureSample(overlay_tex, tex_samp, uv), overlay_straight);
    let pm = over_pm + base_pm * (1.0 - over_pm.a);
    return finalize_pm(pm, out_straight);
}

@fragment fn fs_ss_s(in: VsOut) -> @location(0) vec4<f32> { return composite_impl(in.uv, true, true, true); }
@fragment fn fs_ss_p(in: VsOut) -> @location(0) vec4<f32> { return composite_impl(in.uv, true, true, false); }
@fragment fn fs_sp_s(in: VsOut) -> @location(0) vec4<f32> { return composite_impl(in.uv, true, false, true); }
@fragment fn fs_sp_p(in: VsOut) -> @location(0) vec4<f32> { return composite_impl(in.uv, true, false, false); }
@fragment fn fs_ps_s(in: VsOut) -> @location(0) vec4<f32> { return composite_impl(in.uv, false, true, true); }
@fragment fn fs_ps_p(in: VsOut) -> @location(0) vec4<f32> { return composite_impl(in.uv, false, true, false); }
@fragment fn fs_pp_s(in: VsOut) -> @location(0) vec4<f32> { return composite_impl(in.uv, false, false, true); }
@fragment fn fs_pp_p(in: VsOut) -> @location(0) vec4<f32> { return composite_impl(in.uv, false, false, false); }
"#;

const SINGLE_WGSL: &str = r#"
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

fn finalize_single_impl(uv: vec2<f32>, src_straight: bool, out_straight: bool) -> vec4<f32> {
    let raw = textureSample(src_tex, src_samp, uv);
    let pm = select(raw, vec4<f32>(raw.rgb * raw.a, raw.a), src_straight);
    if (out_straight) {
        let rgb = select(vec3<f32>(0.0), clamp(pm.rgb / pm.a, vec3<f32>(0.0), vec3<f32>(1.0)), pm.a > 1e-5);
        return vec4<f32>(rgb, clamp(pm.a, 0.0, 1.0));
    }
    return pm;
}

@fragment fn fs_s_s(in: VsOut) -> @location(0) vec4<f32> { return finalize_single_impl(in.uv, true, true); }
@fragment fn fs_s_p(in: VsOut) -> @location(0) vec4<f32> { return finalize_single_impl(in.uv, true, false); }
@fragment fn fs_p_s(in: VsOut) -> @location(0) vec4<f32> { return finalize_single_impl(in.uv, false, true); }
@fragment fn fs_p_p(in: VsOut) -> @location(0) vec4<f32> { return finalize_single_impl(in.uv, false, false); }

@fragment
fn fs_stamp_depth(in: VsOut) {
    let c = textureSample(src_tex, src_samp, in.uv);
    if (c.a <= 0.0) {
        discard;
    }
}
"#;

/// Output alpha encoding emitted by [`CompositeFinalizePipeline`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OutputAlphaMode {
    /// Emit straight (unpremultiplied) RGB + A, zeroing RGB when alpha <= 1e-5.
    #[default]
    Straight,
    /// Emit premultiplied RGB + A directly into the destination attachment.
    Premultiplied,
}

/// Errors returned by [`CompositeFinalizePipeline`] depth-stamping operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompositeFinalizeError {
    /// Depth-stamp pass requested on a pipeline created with `depth_format: None`.
    DepthPipelineNotConfigured,
}

impl fmt::Display for CompositeFinalizeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DepthPipelineNotConfigured => {
                write!(f, "CompositeFinalizePipeline created without depth_format")
            }
        }
    }
}

impl std::error::Error for CompositeFinalizeError {}

#[inline]
fn source_mode_idx(mode: SourceAlphaMode) -> usize {
    match mode {
        SourceAlphaMode::Straight => 0,
        SourceAlphaMode::Premultiplied => 1,
    }
}

#[inline]
fn output_mode_idx(mode: OutputAlphaMode) -> usize {
    match mode {
        OutputAlphaMode::Straight => 0,
        OutputAlphaMode::Premultiplied => 1,
    }
}

fn create_fullscreen_color_pipeline(
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
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

fn create_depth_stamp_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    depth_format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("tricore_composite_stamp_depth"),
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
        depth_stencil: Some(wgpu::DepthStencilState {
            format: depth_format,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Always),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_stamp_depth"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[],
        }),
        multiview_mask: None,
        cache: None,
    })
}

fn texture_layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn sampler_layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

/// Pure-WGPU single-pass composite, alpha-finalize (`LoadOp::DontCare`), and overlay depth-stamp pipeline.
pub struct CompositeFinalizePipeline {
    composite_bgl: wgpu::BindGroupLayout,
    single_bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    // Indexed as [base_alpha][overlay_alpha][output_alpha].
    composite_pipelines: [[[wgpu::RenderPipeline; 2]; 2]; 2],
    // Indexed as [src_alpha][output_alpha].
    single_pipelines: [[wgpu::RenderPipeline; 2]; 2],
    depth_stamp_pipeline: Option<wgpu::RenderPipeline>,
}

impl CompositeFinalizePipeline {
    pub fn new(
        device: &wgpu::Device,
        dst_format: wgpu::TextureFormat,
        depth_format: Option<wgpu::TextureFormat>,
    ) -> Self {
        let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tricore_composite_finalize_shader"),
            source: wgpu::ShaderSource::Wgsl(COMPOSITE_WGSL.into()),
        });
        let single_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tricore_single_finalize_shader"),
            source: wgpu::ShaderSource::Wgsl(SINGLE_WGSL.into()),
        });

        let composite_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tricore_composite_finalize_bgl"),
            entries: &[
                texture_layout_entry(0),
                texture_layout_entry(1),
                sampler_layout_entry(2),
            ],
        });
        let single_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tricore_single_finalize_bgl"),
            entries: &[texture_layout_entry(0), sampler_layout_entry(1)],
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("tricore_composite_finalize_sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let composite_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tricore_composite_finalize_layout"),
            bind_group_layouts: &[Some(&composite_bgl)],
            immediate_size: 0,
        });
        let single_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tricore_single_finalize_layout"),
            bind_group_layouts: &[Some(&single_bgl)],
            immediate_size: 0,
        });

        let composite_pipelines = [
            [
                [
                    create_fullscreen_color_pipeline(
                        device,
                        &composite_layout,
                        &composite_shader,
                        dst_format,
                        "fs_ss_s",
                        "tricore_composite_ss_s",
                    ),
                    create_fullscreen_color_pipeline(
                        device,
                        &composite_layout,
                        &composite_shader,
                        dst_format,
                        "fs_ss_p",
                        "tricore_composite_ss_p",
                    ),
                ],
                [
                    create_fullscreen_color_pipeline(
                        device,
                        &composite_layout,
                        &composite_shader,
                        dst_format,
                        "fs_sp_s",
                        "tricore_composite_sp_s",
                    ),
                    create_fullscreen_color_pipeline(
                        device,
                        &composite_layout,
                        &composite_shader,
                        dst_format,
                        "fs_sp_p",
                        "tricore_composite_sp_p",
                    ),
                ],
            ],
            [
                [
                    create_fullscreen_color_pipeline(
                        device,
                        &composite_layout,
                        &composite_shader,
                        dst_format,
                        "fs_ps_s",
                        "tricore_composite_ps_s",
                    ),
                    create_fullscreen_color_pipeline(
                        device,
                        &composite_layout,
                        &composite_shader,
                        dst_format,
                        "fs_ps_p",
                        "tricore_composite_ps_p",
                    ),
                ],
                [
                    create_fullscreen_color_pipeline(
                        device,
                        &composite_layout,
                        &composite_shader,
                        dst_format,
                        "fs_pp_s",
                        "tricore_composite_pp_s",
                    ),
                    create_fullscreen_color_pipeline(
                        device,
                        &composite_layout,
                        &composite_shader,
                        dst_format,
                        "fs_pp_p",
                        "tricore_composite_pp_p",
                    ),
                ],
            ],
        ];

        let single_pipelines = [
            [
                create_fullscreen_color_pipeline(
                    device,
                    &single_layout,
                    &single_shader,
                    dst_format,
                    "fs_s_s",
                    "tricore_single_s_s",
                ),
                create_fullscreen_color_pipeline(
                    device,
                    &single_layout,
                    &single_shader,
                    dst_format,
                    "fs_s_p",
                    "tricore_single_s_p",
                ),
            ],
            [
                create_fullscreen_color_pipeline(
                    device,
                    &single_layout,
                    &single_shader,
                    dst_format,
                    "fs_p_s",
                    "tricore_single_p_s",
                ),
                create_fullscreen_color_pipeline(
                    device,
                    &single_layout,
                    &single_shader,
                    dst_format,
                    "fs_p_p",
                    "tricore_single_p_p",
                ),
            ],
        ];

        let depth_stamp_pipeline =
            depth_format.map(|df| create_depth_stamp_pipeline(device, &single_layout, &single_shader, df));

        Self {
            composite_bgl,
            single_bgl,
            sampler,
            composite_pipelines,
            single_pipelines,
            depth_stamp_pipeline,
        }
    }

    /// Creates a reusable bind group for 2-input `overlay` over `base` compositing.
    pub fn create_composite_bind_group(
        &self,
        device: &wgpu::Device,
        base_view: &wgpu::TextureView,
        overlay_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tricore_composite_finalize_bg"),
            layout: &self.composite_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(base_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(overlay_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        })
    }

    /// Creates a reusable bind group for 1-input finalization or overlay depth stamping.
    pub fn create_single_bind_group(
        &self,
        device: &wgpu::Device,
        src_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tricore_single_finalize_bg"),
            layout: &self.single_bgl,
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

    fn run_fullscreen_color_pass(
        encoder: &mut wgpu::CommandEncoder,
        dst_view: &wgpu::TextureView,
        pipeline: &wgpu::RenderPipeline,
        bind_group: &wgpu::BindGroup,
        label: &'static str,
    ) {
        // SAFETY: the fullscreen triangle covers the entire viewport with color blending disabled.
        let load = wgpu::LoadOp::DontCare(unsafe { wgpu::LoadOpDontCare::enabled() });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: dst_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load,
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

    /// Composites `overlay` over `base` via `f32` Porter-Duff `SrcOver` into `dst_view` (`LoadOp::DontCare`).
    pub fn composite_with_bind_group(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        dst_view: &wgpu::TextureView,
        bind_group: &wgpu::BindGroup,
        base_alpha: SourceAlphaMode,
        overlay_alpha: SourceAlphaMode,
        output_alpha: OutputAlphaMode,
    ) {
        let pipeline = &self.composite_pipelines[source_mode_idx(base_alpha)]
            [source_mode_idx(overlay_alpha)][output_mode_idx(output_alpha)];
        Self::run_fullscreen_color_pass(
            encoder,
            dst_view,
            pipeline,
            bind_group,
            "tricore_composite_finalize_pass",
        );
    }

    /// Convenience wrapper creating a temporary composite bind group and encoding [`Self::composite_with_bind_group`].
    #[allow(clippy::too_many_arguments)]
    pub fn composite(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        dst_view: &wgpu::TextureView,
        base_view: &wgpu::TextureView,
        overlay_view: &wgpu::TextureView,
        base_alpha: SourceAlphaMode,
        overlay_alpha: SourceAlphaMode,
        output_alpha: OutputAlphaMode,
    ) {
        let bg = self.create_composite_bind_group(device, base_view, overlay_view);
        self.composite_with_bind_group(encoder, dst_view, &bg, base_alpha, overlay_alpha, output_alpha);
    }

    /// Finalizes a single source texture into `dst_view` (`LoadOp::DontCare`).
    pub fn finalize_single_with_bind_group(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        dst_view: &wgpu::TextureView,
        bind_group: &wgpu::BindGroup,
        src_alpha: SourceAlphaMode,
        output_alpha: OutputAlphaMode,
    ) {
        let pipeline =
            &self.single_pipelines[source_mode_idx(src_alpha)][output_mode_idx(output_alpha)];
        Self::run_fullscreen_color_pass(
            encoder,
            dst_view,
            pipeline,
            bind_group,
            "tricore_single_finalize_pass",
        );
    }

    /// Convenience wrapper creating a temporary bind group and encoding [`Self::finalize_single_with_bind_group`].
    pub fn finalize_single(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        dst_view: &wgpu::TextureView,
        src_view: &wgpu::TextureView,
        src_alpha: SourceAlphaMode,
        output_alpha: OutputAlphaMode,
    ) {
        let bg = self.create_single_bind_group(device, src_view);
        self.finalize_single_with_bind_group(encoder, dst_view, &bg, src_alpha, output_alpha);
    }

    /// Stamps `z = 0.0` into `depth_view` (`LoadOp::Load`) wherever the bound overlay texture has `alpha > 0.0`.
    pub fn stamp_overlay_depth_with_bind_group(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        depth_view: &wgpu::TextureView,
        single_bind_group: &wgpu::BindGroup,
    ) -> Result<(), CompositeFinalizeError> {
        let pipeline = self
            .depth_stamp_pipeline
            .as_ref()
            .ok_or(CompositeFinalizeError::DepthPipelineNotConfigured)?;
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("tricore_composite_stamp_depth_pass"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, single_bind_group, &[]);
        pass.draw(0..3, 0..1);
        Ok(())
    }

    /// Convenience wrapper creating a temporary single-texture bind group and stamping `z = 0.0` into `depth_view`.
    pub fn stamp_overlay_depth(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        depth_view: &wgpu::TextureView,
        overlay_view: &wgpu::TextureView,
    ) -> Result<(), CompositeFinalizeError> {
        let bg = self.create_single_bind_group(device, overlay_view);
        self.stamp_overlay_depth_with_bind_group(encoder, depth_view, &bg)
    }
}
