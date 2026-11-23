//! CPU readback buffers for color and depth buffers (essential for SketchUp C Overlay and test harnesses).

use std::fmt;

/// Channel swizzle mode applied during [`ReadbackSurface`] row repacking.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ChannelSwizzle {
    /// Preserve source texture channel byte order (e.g., RGBA->RGBA or BGRA->BGRA).
    #[default]
    None,
    /// Swap red and blue channels (e.g., RGBA->BGRA or BGRA->RGBA).
    SwapRB,
}

/// Configuration for [`ReadbackSurface`] row repacking and channel conversions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReadbackOptions {
    pub swizzle: ChannelSwizzle,
    pub y_flip: bool,
    pub unpremultiply: bool,
}

/// Configuration for [`DepthReadbackSurface`] row repacking.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DepthReadbackOptions {
    pub y_flip: bool,
}

/// Errors produced by [`ReadbackSurface`] and [`DepthReadbackSurface`] map/poll operations.
#[derive(Debug)]
pub enum ReadbackError {
    NotInitialized,
    Poll(wgpu::PollError),
    ChannelClosed(std::sync::mpsc::RecvError),
    BufferAsync(wgpu::BufferAsyncError),
    MapRange(wgpu::MapRangeError),
}

impl fmt::Display for ReadbackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotInitialized => write!(f, "readback staging buffer is not initialized"),
            Self::Poll(err) => write!(f, "device poll failed during readback: {err}"),
            Self::ChannelClosed(err) => write!(f, "buffer map callback channel closed: {err}"),
            Self::BufferAsync(err) => write!(f, "buffer map async operation failed: {err}"),
            Self::MapRange(err) => write!(f, "failed to map staging buffer range: {err}"),
        }
    }
}

impl std::error::Error for ReadbackError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::NotInitialized => None,
            Self::Poll(err) => Some(err),
            Self::ChannelClosed(err) => Some(err),
            Self::BufferAsync(err) => Some(err),
            Self::MapRange(err) => Some(err),
        }
    }
}

fn copy_texture_to_staging_buffer(
    encoder: &mut wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    aspect: wgpu::TextureAspect,
    buffer: &wgpu::Buffer,
    width: u32,
    height: u32,
    row_pitch: u32,
) {
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect,
        },
        wgpu::TexelCopyBufferInfo {
            buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_pitch),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
}

fn map_buffer_sync(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    on_mapped: impl FnOnce(&[u8]),
) -> Result<(), ReadbackError> {
    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });

    #[cfg(not(target_arch = "wasm32"))]
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(ReadbackError::Poll)?;

    rx.recv()
        .map_err(ReadbackError::ChannelClosed)?
        .map_err(ReadbackError::BufferAsync)?;

    {
        let mapped = slice.get_mapped_range().map_err(ReadbackError::MapRange)?;
        on_mapped(&mapped);
    }
    buffer.unmap();
    Ok(())
}

/// Buffer for reading rendered RGBA/BGRA color pixels from GPU memory into CPU host memory.
pub struct ColorReadbackBuffer {
    pub buffer: wgpu::Buffer,
    pub width: u32,
    pub height: u32,
    pub row_pitch: u32,
    pub unpadded_row_bytes: u32,
}

impl ColorReadbackBuffer {
    /// Creates a staging buffer matching the required alignment for texture copy (256 bytes aligned).
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let unpadded_row_bytes = width * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let row_pitch = unpadded_row_bytes.next_multiple_of(align);
        let buffer_size = (row_pitch * height) as u64;

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tricore_color_readback_buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            buffer,
            width,
            height,
            row_pitch,
            unpadded_row_bytes,
        }
    }

    /// Enqueues a copy from the offscreen color texture into this readback staging buffer.
    pub fn copy_from_texture(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        texture: &wgpu::Texture,
    ) {
        copy_texture_to_staging_buffer(
            encoder,
            texture,
            wgpu::TextureAspect::All,
            &self.buffer,
            self.width,
            self.height,
            self.row_pitch,
        );
    }
}

/// Reusable GPU-to-CPU color readback surface owning a staging buffer and a persistent pixel cache.
///
/// Handles 256-byte row-pitch unpadding, optional R/B channel swizzling, optional Y-axis flip,
/// and optional unpremultiplication, while retaining [`Self::cached_pixels`] for zero-cost idle frames.
pub struct ReadbackSurface {
    staging: Option<ColorReadbackBuffer>,
    cached_pixels: Vec<u8>,
    width: u32,
    height: u32,
    options: ReadbackOptions,
}

impl ReadbackSurface {
    pub fn new(options: ReadbackOptions) -> Self {
        Self {
            staging: None,
            cached_pixels: Vec::new(),
            width: 0,
            height: 0,
            options,
        }
    }

    pub fn options(&self) -> ReadbackOptions {
        self.options
    }

    pub fn set_options(&mut self, options: ReadbackOptions) {
        self.options = options;
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn unpadded_row_pitch(&self) -> u32 {
        self.width * 4
    }

    pub fn size_bytes(&self) -> usize {
        self.cached_pixels.len()
    }

    pub fn cached_pixels(&self) -> &[u8] {
        &self.cached_pixels
    }

    /// Ensures the staging buffer and CPU cache match `(width, height)`, clamping to at least 1x1.
    /// Returns `true` if the staging buffer was allocated or reallocated.
    pub fn ensure_dimensions(&mut self, device: &wgpu::Device, width: u32, height: u32) -> bool {
        let w = width.max(1);
        let h = height.max(1);
        if self.staging.is_some() && self.width == w && self.height == h {
            return false;
        }
        self.staging = Some(ColorReadbackBuffer::new(device, w, h));
        self.cached_pixels.resize((w * h * 4) as usize, 0);
        self.width = w;
        self.height = h;
        true
    }

    /// Encodes a GPU texture-to-staging-buffer copy into `encoder`.
    pub fn copy_from_texture(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        texture: &wgpu::Texture,
    ) -> Result<(), ReadbackError> {
        let staging = self.staging.as_ref().ok_or(ReadbackError::NotInitialized)?;
        staging.copy_from_texture(encoder, texture);
        Ok(())
    }

    /// Synchronously maps the staging buffer, unpacks rows into [`Self::cached_pixels`], and unmaps.
    pub fn map_and_repack(&mut self, device: &wgpu::Device) -> Result<&[u8], ReadbackError> {
        let staging = self.staging.as_ref().ok_or(ReadbackError::NotInitialized)?;
        let width = self.width;
        let height = self.height;
        let row_pitch = staging.row_pitch;
        let options = self.options;
        let cached = &mut self.cached_pixels;

        map_buffer_sync(device, &staging.buffer, |mapped| {
            repack_rgba_rows(mapped, cached, width, height, row_pitch, options);
        })?;
        Ok(&self.cached_pixels)
    }

    /// Ensures dimensions match `texture`, submits a copy command, and reads back the pixels synchronously.
    pub fn readback_sync(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
    ) -> Result<&[u8], ReadbackError> {
        self.ensure_dimensions(device, texture.width(), texture.height());
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("tricore_readback_encoder"),
        });
        self.copy_from_texture(&mut encoder, texture)?;
        queue.submit(std::iter::once(encoder.finish()));
        self.map_and_repack(device)
    }
}

/// Repacks 256-byte padded 4-channel GPU rows into tightly packed CPU pixels with optional
/// Y-flip, unpremultiplication, and R/B channel swizzling.
pub fn repack_rgba_rows(
    src_padded: &[u8],
    dst: &mut [u8],
    width: u32,
    height: u32,
    padded_row_pitch: u32,
    options: ReadbackOptions,
) {
    let w = width as usize;
    let h = height as usize;
    let unpadded = w * 4;
    let padded = padded_row_pitch as usize;
    debug_assert!(src_padded.len() >= h.saturating_sub(1) * padded + unpadded);
    debug_assert!(dst.len() >= h * unpadded);

    for y in 0..h {
        let dst_y = if options.y_flip { h - 1 - y } else { y };
        let src_row = &src_padded[y * padded..y * padded + unpadded];
        let dst_row = &mut dst[dst_y * unpadded..dst_y * unpadded + unpadded];

        if options.swizzle == ChannelSwizzle::None && !options.unpremultiply {
            dst_row.copy_from_slice(src_row);
            continue;
        }

        for (src_px, dst_px) in src_row.chunks_exact(4).zip(dst_row.chunks_exact_mut(4)) {
            let (mut c0, mut c1, mut c2, a) = (src_px[0], src_px[1], src_px[2], src_px[3]);
            if options.unpremultiply {
                if a == 0 {
                    c0 = 0;
                    c1 = 0;
                    c2 = 0;
                } else if a < 255 {
                    let a32 = a as u32;
                    c0 = ((c0 as u32 * 255 + a32 / 2) / a32).min(255) as u8;
                    c1 = ((c1 as u32 * 255 + a32 / 2) / a32).min(255) as u8;
                    c2 = ((c2 as u32 * 255 + a32 / 2) / a32).min(255) as u8;
                }
            }

            match options.swizzle {
                ChannelSwizzle::None => {
                    dst_px[0] = c0;
                    dst_px[1] = c1;
                    dst_px[2] = c2;
                    dst_px[3] = a;
                }
                ChannelSwizzle::SwapRB => {
                    dst_px[0] = c2;
                    dst_px[1] = c1;
                    dst_px[2] = c0;
                    dst_px[3] = a;
                }
            }
        }
    }
}

/// Buffer for reading rendered depth values (`Depth32Float`) from GPU memory into CPU host memory.
pub struct DepthReadbackBuffer {
    pub buffer: wgpu::Buffer,
    pub width: u32,
    pub height: u32,
    pub row_pitch: u32,
    pub unpadded_row_bytes: u32,
}

impl DepthReadbackBuffer {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let unpadded_row_bytes = width * std::mem::size_of::<f32>() as u32;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let row_pitch = unpadded_row_bytes.next_multiple_of(align);
        let buffer_size = (row_pitch * height) as u64;

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tricore_depth_readback_buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            buffer,
            width,
            height,
            row_pitch,
            unpadded_row_bytes,
        }
    }

    /// Enqueues a `DepthOnly` copy from the `Depth32Float` texture into this staging buffer.
    pub fn copy_from_texture(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        texture: &wgpu::Texture,
    ) {
        copy_texture_to_staging_buffer(
            encoder,
            texture,
            wgpu::TextureAspect::DepthOnly,
            &self.buffer,
            self.width,
            self.height,
            self.row_pitch,
        );
    }
}

/// Reusable GPU-to-CPU `Depth32Float` readback surface owning a staging buffer and persistent `[0.0, 1.0]` cache.
pub struct DepthReadbackSurface {
    staging: Option<DepthReadbackBuffer>,
    cached_depths: Vec<f32>,
    width: u32,
    height: u32,
    options: DepthReadbackOptions,
}

impl DepthReadbackSurface {
    pub fn new(options: DepthReadbackOptions) -> Self {
        Self {
            staging: None,
            cached_depths: Vec::new(),
            width: 0,
            height: 0,
            options,
        }
    }

    pub fn options(&self) -> DepthReadbackOptions {
        self.options
    }

    pub fn set_options(&mut self, options: DepthReadbackOptions) {
        self.options = options;
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn cached_depths(&self) -> &[f32] {
        &self.cached_depths
    }

    pub fn cached_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.cached_depths)
    }

    /// Ensures the depth staging buffer and CPU cache match `(width, height)`, clamping to at least 1x1.
    /// Returns `true` if the staging buffer was allocated or reallocated.
    pub fn ensure_dimensions(&mut self, device: &wgpu::Device, width: u32, height: u32) -> bool {
        let w = width.max(1);
        let h = height.max(1);
        if self.staging.is_some() && self.width == w && self.height == h {
            return false;
        }
        self.staging = Some(DepthReadbackBuffer::new(device, w, h));
        self.cached_depths.resize((w * h) as usize, 1.0);
        self.width = w;
        self.height = h;
        true
    }

    /// Encodes a `DepthOnly` texture-to-staging-buffer copy into `encoder`.
    pub fn copy_from_texture(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        depth_texture: &wgpu::Texture,
    ) -> Result<(), ReadbackError> {
        let staging = self.staging.as_ref().ok_or(ReadbackError::NotInitialized)?;
        staging.copy_from_texture(encoder, depth_texture);
        Ok(())
    }

    /// Synchronously maps the staging buffer, unpacks depth rows into [`Self::cached_depths`], and unmaps.
    pub fn map_and_repack(&mut self, device: &wgpu::Device) -> Result<&[f32], ReadbackError> {
        let staging = self.staging.as_ref().ok_or(ReadbackError::NotInitialized)?;
        let width = self.width;
        let height = self.height;
        let row_pitch = staging.row_pitch;
        let options = self.options;
        let cached = &mut self.cached_depths;

        map_buffer_sync(device, &staging.buffer, |mapped| {
            repack_depth_rows(mapped, cached, width, height, row_pitch, options);
        })?;
        Ok(&self.cached_depths)
    }

    /// Ensures dimensions match `depth_texture`, submits a copy command, and reads back depths synchronously.
    pub fn readback_sync(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        depth_texture: &wgpu::Texture,
    ) -> Result<&[f32], ReadbackError> {
        self.ensure_dimensions(device, depth_texture.width(), depth_texture.height());
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("tricore_depth_readback_encoder"),
        });
        self.copy_from_texture(&mut encoder, depth_texture)?;
        queue.submit(std::iter::once(encoder.finish()));
        self.map_and_repack(device)
    }
}

/// Repacks 256-byte padded `Depth32Float` GPU rows into tightly packed `f32` slices with optional Y-flip.
pub fn repack_depth_rows(
    src_padded: &[u8],
    dst: &mut [f32],
    width: u32,
    height: u32,
    padded_row_pitch: u32,
    options: DepthReadbackOptions,
) {
    let w = width as usize;
    let h = height as usize;
    let unpadded_bytes = w * std::mem::size_of::<f32>();
    let padded = padded_row_pitch as usize;
    debug_assert!(src_padded.len() >= h.saturating_sub(1) * padded + unpadded_bytes);
    debug_assert!(dst.len() >= h * w);

    for y in 0..h {
        let dst_y = if options.y_flip { h - 1 - y } else { y };
        let src_bytes = &src_padded[y * padded..y * padded + unpadded_bytes];
        let dst_row = &mut dst[dst_y * w..dst_y * w + w];
        bytemuck::cast_slice_mut(dst_row).copy_from_slice(src_bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repack_unpads_swizzles_flips_and_unpremultiplies() {
        let width = 2;
        let height = 2;
        let pitch = 256;
        let mut padded = vec![0u8; pitch * height];

        // Row 0: pixel(0,0) = [64, 32, 0, 128] (premul), pixel(1,0) = [10, 20, 30, 255]
        padded[0..4].copy_from_slice(&[64, 32, 0, 128]);
        padded[4..8].copy_from_slice(&[10, 20, 30, 255]);
        // Row 1: pixel(0,1) = [99, 88, 77, 0], pixel(1,1) = [255, 128, 64, 255]
        padded[pitch..pitch + 4].copy_from_slice(&[99, 88, 77, 0]);
        padded[pitch + 4..pitch + 8].copy_from_slice(&[255, 128, 64, 255]);

        let mut dst = vec![0u8; width * height * 4];
        repack_rgba_rows(
            &padded,
            &mut dst,
            width as u32,
            height as u32,
            pitch as u32,
            ReadbackOptions {
                swizzle: ChannelSwizzle::SwapRB,
                y_flip: true,
                unpremultiply: true,
            },
        );

        // Flipped: Row 1 becomes row 0, Row 0 becomes row 1.
        // Old (0,1) had alpha=0 -> zeroed RGB
        assert_eq!(&dst[0..4], &[0, 0, 0, 0]);
        // Old (1,1) -> swapped [64, 128, 255, 255]
        assert_eq!(&dst[4..8], &[64, 128, 255, 255]);
        // Old (0,0) unpremultiplied: r = (64*255+64)/128 = 128, g = (32*255+64)/128 = 64, b = 0 -> swapped [0, 64, 128, 128]
        assert_eq!(&dst[8..12], &[0, 64, 128, 128]);
        // Old (1,0) -> swapped [30, 20, 10, 255]
        assert_eq!(&dst[12..16], &[30, 20, 10, 255]);

        // Fast-path ChannelSwizzle::None without unpremultiply preserves exact byte order.
        let mut fast_dst = vec![0u8; width * height * 4];
        repack_rgba_rows(
            &padded,
            &mut fast_dst,
            width as u32,
            height as u32,
            pitch as u32,
            ReadbackOptions {
                swizzle: ChannelSwizzle::None,
                y_flip: false,
                unpremultiply: false,
            },
        );
        assert_eq!(&fast_dst[0..4], &[64, 32, 0, 128]);
        assert_eq!(&fast_dst[4..8], &[10, 20, 30, 255]);
        assert_eq!(&fast_dst[8..12], &[99, 88, 77, 0]);
        assert_eq!(&fast_dst[12..16], &[255, 128, 64, 255]);
    }

    #[test]
    fn repack_depth_rows_unpads_and_flips() {
        let width = 3;
        let height = 2;
        let pitch = 256;
        let mut padded = vec![0u8; pitch * height];

        let row0 = [0.1_f32, 0.25_f32, 0.5_f32];
        let row1 = [0.75_f32, 0.9_f32, 1.0_f32];
        padded[0..12].copy_from_slice(bytemuck::cast_slice(&row0));
        padded[pitch..pitch + 12].copy_from_slice(bytemuck::cast_slice(&row1));

        let mut dst_no_flip = vec![0.0_f32; width * height];
        repack_depth_rows(
            &padded,
            &mut dst_no_flip,
            width as u32,
            height as u32,
            pitch as u32,
            DepthReadbackOptions { y_flip: false },
        );
        assert_eq!(dst_no_flip, vec![0.1, 0.25, 0.5, 0.75, 0.9, 1.0]);

        let mut dst_flip = vec![0.0_f32; width * height];
        repack_depth_rows(
            &padded,
            &mut dst_flip,
            width as u32,
            height as u32,
            pitch as u32,
            DepthReadbackOptions { y_flip: true },
        );
        assert_eq!(dst_flip, vec![0.75, 0.9, 1.0, 0.1, 0.25, 0.5]);
    }
}
