//! WGPU GPU context and device/queue lifecycle orchestration.

use std::fmt;
use std::sync::Arc;

/// Error returned when headless WGPU initialization fails.
#[derive(Debug)]
pub enum GpuInitError {
    NoAdapter(wgpu::RequestAdapterError),
    RequestDevice(wgpu::RequestDeviceError),
}

impl fmt::Display for GpuInitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoAdapter(err) => write!(f, "failed to acquire WGPU adapter: {err}"),
            Self::RequestDevice(err) => write!(f, "failed to create WGPU device: {err}"),
        }
    }
}

impl std::error::Error for GpuInitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::NoAdapter(err) => Some(err),
            Self::RequestDevice(err) => Some(err),
        }
    }
}

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

    /// Returns the preferred native WGPU backend flags for the current host OS.
    pub fn platform_backends() -> wgpu::Backends {
        #[cfg(target_os = "macos")]
        {
            wgpu::Backends::METAL
        }
        #[cfg(target_os = "windows")]
        {
            wgpu::Backends::DX12 | wgpu::Backends::VULKAN
        }
        #[cfg(not(any(target_os = "macos", target_os = "windows")))]
        {
            wgpu::Backends::all()
        }
    }

    /// Initializes a standalone headless GPU context using the host platform's preferred backends.
    pub async fn new_headless_for_platform(
        power_preference: wgpu::PowerPreference,
    ) -> Result<Self, GpuInitError> {
        Self::new_headless_with_backends(Self::platform_backends(), power_preference).await
    }

    /// Initializes a standalone headless GPU context using explicit WGPU backends,
    /// falling back to a software/CPU adapter if hardware adapter selection fails.
    pub async fn new_headless_with_backends(
        backends: wgpu::Backends,
        power_preference: wgpu::PowerPreference,
    ) -> Result<Self, GpuInitError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = match instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference,
                force_fallback_adapter: false,
                compatible_surface: None,
                ..Default::default()
            })
            .await
        {
            Ok(adapter) => adapter,
            Err(_) => instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::LowPower,
                    force_fallback_adapter: true,
                    compatible_surface: None,
                    ..Default::default()
                })
                .await
                .map_err(GpuInitError::NoAdapter)?,
        };

        let adapter_info = adapter.get_info();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("tricore_headless_device"),
                ..Default::default()
            })
            .await
            .map_err(GpuInitError::RequestDevice)?;

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter_info: Some(adapter_info),
        })
    }

    /// Initializes a standalone headless GPU context for offscreen rendering.
    pub async fn new_headless(
        power_preference: wgpu::PowerPreference,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Self::new_headless_with_backends(wgpu::Backends::all(), power_preference)
            .await
            .map_err(Into::into)
    }

    /// Returns `true` when the active adapter is a CPU/software fallback device.
    pub fn is_software_fallback(&self) -> bool {
        self.adapter_info
            .as_ref()
            .is_some_and(|info| info.device_type == wgpu::DeviceType::Cpu)
    }
}
