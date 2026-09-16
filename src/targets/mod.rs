//! Render target abstractions supporting both surface presentation and offscreen rendering.

/// Dimensions of a render target including device pixel ratio scale factor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TargetDimensions {
    pub width: u32,
    pub height: u32,
    pub scale_factor: f64,
}

impl TargetDimensions {
    pub fn new(width: u32, height: u32, scale_factor: f64) -> Self {
        Self {
            width: width.max(1),
            height: height.max(1),
            scale_factor: if scale_factor > 0.0 { scale_factor } else { 1.0 },
        }
    }

    /// Physical pixel width.
    pub fn physical_width(&self) -> u32 {
        ((self.width as f64) * self.scale_factor).round() as u32
    }

    /// Physical pixel height.
    pub fn physical_height(&self) -> u32 {
        ((self.height as f64) * self.scale_factor).round() as u32
    }
}

/// An offscreen render target owning a color texture and an optional depth texture.
pub struct OffscreenTarget {
    pub dimensions: TargetDimensions,
    pub color_texture: wgpu::Texture,
    pub color_view: wgpu::TextureView,
    pub color_format: wgpu::TextureFormat,
    pub depth_texture: Option<wgpu::Texture>,
    pub depth_view: Option<wgpu::TextureView>,
    pub depth_format: Option<wgpu::TextureFormat>,
}

impl OffscreenTarget {
    /// Creates a new offscreen render target with the specified dimensions and formats.
    pub fn new(
        device: &wgpu::Device,
        dimensions: TargetDimensions,
        color_format: wgpu::TextureFormat,
        depth_format: Option<wgpu::TextureFormat>,
    ) -> Self {
        let phys_w = dimensions.physical_width();
        let phys_h = dimensions.physical_height();

        let color_desc = wgpu::TextureDescriptor {
            label: Some("tricore_offscreen_color"),
            size: wgpu::Extent3d {
                width: phys_w,
                height: phys_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: color_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let color_texture = device.create_texture(&color_desc);
        let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let (depth_texture, depth_view) = if let Some(df) = depth_format {
            let depth_desc = wgpu::TextureDescriptor {
                label: Some("tricore_offscreen_depth"),
                size: wgpu::Extent3d {
                    width: phys_w,
                    height: phys_h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: df,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            };
            let tex = device.create_texture(&depth_desc);
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            (Some(tex), Some(view))
        } else {
            (None, None)
        };

        Self {
            dimensions,
            color_texture,
            color_view,
            color_format,
            depth_texture,
            depth_view,
            depth_format,
        }
    }
}
