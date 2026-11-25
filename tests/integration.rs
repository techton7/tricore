use std::sync::OnceLock;
use tricore::GpuContext;

static SHARED_GPU: OnceLock<Option<GpuContext>> = OnceLock::new();

pub fn try_shared_gpu() -> Option<&'static GpuContext> {
    SHARED_GPU
        .get_or_init(|| {
            pollster::block_on(GpuContext::new_headless_for_platform(
                wgpu::PowerPreference::LowPower,
            ))
            .ok()
        })
        .as_ref()
}

pub fn shared_gpu() -> &'static GpuContext {
    try_shared_gpu().expect("headless WGPU adapter required for tricore integration tests")
}

mod depth_peeling;
mod pipelines;
mod targets_readback;
