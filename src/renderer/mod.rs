//! WGPU GPU context and device/queue lifecycle orchestration.

use std::sync::Arc;

/// Shared GPU context wrapping `wgpu::Device` and `wgpu::Queue`.
///
/// Can either own a standalone headless device or adopt external Device/Queue handles
/// provided by host engines such as Blitz Native or SketchUp plugins.
#[derive(Clone)]
pub struct GpuContext {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub adapter_info: Option<wgpu::AdapterInfo>,
}

impl GpuContext {
    /// Creates a GpuContext by adopting existing Device and Queue handles from a host.
    pub fn from_shared(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        Self {
            device,
            queue,
            adapter_info: None,
        }
    }

    /// Initializes a standalone headless GPU context for offscreen rendering.
    pub async fn new_headless(
        power_preference: wgpu::PowerPreference,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await?;

        let adapter_info = adapter.get_info();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("tricore_headless_device"),
                ..Default::default()
            })
            .await?;

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter_info: Some(adapter_info),
        })
    }
}
