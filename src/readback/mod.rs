//! CPU readback buffers for color and depth buffers (essential for SketchUp C Overlay and test harnesses).

/// Buffer for reading rendered RGBA color pixels from GPU memory into CPU host memory.
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
        let row_pitch = (unpadded_row_bytes + align - 1) / align * align;
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

    /// Enqueues a copy from the offscreen texture into this readback staging buffer.
    pub fn copy_from_texture(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        texture: &wgpu::Texture,
    ) {
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.row_pitch),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
    }
}

/// Buffer for reading rendered depth values (Depth32Float) from GPU memory into CPU host memory.
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
        let row_pitch = (unpadded_row_bytes + align - 1) / align * align;
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
}
