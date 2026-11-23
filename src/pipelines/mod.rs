//! Pure-WGPU render pipelines with zero external UI or gizmo dependencies.

pub mod alpha_blit;
pub mod gizmo2d;
pub mod preview3d;

pub use alpha_blit::*;
pub use gizmo2d::*;
pub use preview3d::*;

/// Growable GPU buffer helper shared across dynamic batching pipelines.
pub(crate) struct DynamicGpuBuffer {
    pub buffer: wgpu::Buffer,
    capacity: u64,
    usage: wgpu::BufferUsages,
    label: &'static str,
}

impl DynamicGpuBuffer {
    pub fn new(
        device: &wgpu::Device,
        initial_bytes: u64,
        usage: wgpu::BufferUsages,
        label: &'static str,
    ) -> Self {
        let capacity = initial_bytes.max(256).next_power_of_two();
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: capacity,
            usage: usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            buffer,
            capacity,
            usage: usage | wgpu::BufferUsages::COPY_DST,
            label,
        }
    }

    pub fn write_pod<T: bytemuck::Pod>(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &[T],
    ) {
        let bytes = bytemuck::cast_slice(data);
        if bytes.is_empty() {
            return;
        }
        let required = bytes.len() as u64;
        if required > self.capacity {
            self.capacity = required.max(256).next_power_of_two();
            self.buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(self.label),
                size: self.capacity,
                usage: self.usage,
                mapped_at_creation: false,
            });
        }
        queue.write_buffer(&self.buffer, 0, bytes);
    }
}
