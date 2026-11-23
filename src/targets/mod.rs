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

/// Configuration options for [`OffscreenTarget`] texture creation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OffscreenTargetOptions {
    pub color_usage: wgpu::TextureUsages,
    pub depth_usage: wgpu::TextureUsages,
}

impl Default for OffscreenTargetOptions {
    fn default() -> Self {
        Self {
            color_usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            depth_usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        }
    }
}

impl OffscreenTargetOptions {
    /// Returns options whose color texture includes `STORAGE_BINDING` (required by compute renderers like Vello).
    pub fn storage_compatible() -> Self {
        let mut opts = Self::default();
        opts.color_usage |= wgpu::TextureUsages::STORAGE_BINDING;
        opts
    }

    /// Returns default options for `color_format`, enabling `STORAGE_BINDING` for `Rgba8Unorm`.
    pub fn for_format(color_format: wgpu::TextureFormat) -> Self {
        if matches!(color_format, wgpu::TextureFormat::Rgba8Unorm) {
            Self::storage_compatible()
        } else {
            Self::default()
        }
    }
}

/// An offscreen render target owning a color texture and an optional depth texture.
pub struct OffscreenTarget {
    pub dimensions: TargetDimensions,
    pub color_texture: wgpu::Texture,
    pub color_view: wgpu::TextureView,
    pub color_format: wgpu::TextureFormat,
    pub color_usage: wgpu::TextureUsages,
    pub depth_texture: Option<wgpu::Texture>,
    pub depth_view: Option<wgpu::TextureView>,
    pub depth_format: Option<wgpu::TextureFormat>,
    pub depth_usage: wgpu::TextureUsages,
}

impl OffscreenTarget {
    /// Returns the default color texture usages for a given format.
    pub fn default_color_usages(format: wgpu::TextureFormat) -> wgpu::TextureUsages {
        OffscreenTargetOptions::for_format(format).color_usage
    }

    /// Creates a new offscreen render target with default usages for `color_format`.
    pub fn new(
        device: &wgpu::Device,
        dimensions: TargetDimensions,
        color_format: wgpu::TextureFormat,
        depth_format: Option<wgpu::TextureFormat>,
    ) -> Self {
        Self::new_with_options(
            device,
            dimensions,
            color_format,
            depth_format,
            OffscreenTargetOptions::for_format(color_format),
        )
    }

    /// Creates a new offscreen render target with an explicit color texture usage mask.
    pub fn new_with_usage(
        device: &wgpu::Device,
        dimensions: TargetDimensions,
        color_format: wgpu::TextureFormat,
        depth_format: Option<wgpu::TextureFormat>,
        color_usage: wgpu::TextureUsages,
    ) -> Self {
        Self::new_with_options(
            device,
            dimensions,
            color_format,
            depth_format,
            OffscreenTargetOptions {
                color_usage,
                ..Default::default()
            },
        )
    }

    /// Creates a new offscreen render target with `STORAGE_BINDING` enabled on the color texture.
    pub fn new_storage_compatible(
        device: &wgpu::Device,
        dimensions: TargetDimensions,
        color_format: wgpu::TextureFormat,
        depth_format: Option<wgpu::TextureFormat>,
    ) -> Self {
        Self::new_with_options(
            device,
            dimensions,
            color_format,
            depth_format,
            OffscreenTargetOptions::storage_compatible(),
        )
    }

    /// Creates a new offscreen render target with full [`OffscreenTargetOptions`].
    pub fn new_with_options(
        device: &wgpu::Device,
        dimensions: TargetDimensions,
        color_format: wgpu::TextureFormat,
        depth_format: Option<wgpu::TextureFormat>,
        options: OffscreenTargetOptions,
    ) -> Self {
        let phys_w = dimensions.physical_width().max(1);
        let phys_h = dimensions.physical_height().max(1);

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
            usage: options.color_usage,
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
                usage: options.depth_usage,
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
            color_usage: options.color_usage,
            depth_texture,
            depth_view,
            depth_format,
            depth_usage: options.depth_usage,
        }
    }

    /// Recreates the underlying textures only when physical pixel dimensions change,
    /// preserving the target's formats and usage flags. Returns `true` if reallocated.
    pub fn ensure_dimensions(
        &mut self,
        device: &wgpu::Device,
        dimensions: TargetDimensions,
    ) -> bool {
        let new_w = dimensions.physical_width().max(1);
        let new_h = dimensions.physical_height().max(1);
        if self.dimensions.physical_width().max(1) == new_w
            && self.dimensions.physical_height().max(1) == new_h
        {
            self.dimensions = dimensions;
            return false;
        }
        *self = Self::new_with_options(
            device,
            dimensions,
            self.color_format,
            self.depth_format,
            OffscreenTargetOptions {
                color_usage: self.color_usage,
                depth_usage: self.depth_usage,
            },
        );
        true
    }
}
