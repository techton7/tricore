//! Renderer-owned handles and controllers for smooth motion interpolation (kinetocore / kinetoxus).

use std::sync::{Arc, RwLock};
use glam::Vec3;

/// Thread-safe animatable camera controller handle.
#[derive(Clone)]
pub struct CameraHandle {
    inner: Arc<RwLock<CameraState>>,
}

#[derive(Clone, Copy, Debug)]
pub struct CameraState {
    pub position: Vec3,
    pub target: Vec3,
    pub zoom: f32,
}

impl Default for CameraHandle {
    fn default() -> Self {
        Self::new(Vec3::new(0.0, 5.0, 10.0), Vec3::ZERO, 1.0)
    }
}

impl CameraHandle {
    pub fn new(position: Vec3, target: Vec3, zoom: f32) -> Self {
        Self {
            inner: Arc::new(RwLock::new(CameraState {
                position,
                target,
                zoom,
            })),
        }
    }

    pub fn get(&self) -> CameraState {
        *self.inner.read().unwrap()
    }

    pub fn set_position(&self, pos: Vec3) {
        self.inner.write().unwrap().position = pos;
    }

    pub fn set_target(&self, target: Vec3) {
        self.inner.write().unwrap().target = target;
    }

    pub fn set_zoom(&self, zoom: f32) {
        self.inner.write().unwrap().zoom = zoom;
    }
}

/// Generic numeric array target for in-place tweening by motion engines.
#[derive(Clone)]
pub struct FloatBufferHandle {
    data: Arc<RwLock<Vec<f32>>>,
}

impl FloatBufferHandle {
    pub fn new(initial: Vec<f32>) -> Self {
        Self {
            data: Arc::new(RwLock::new(initial)),
        }
    }

    pub fn read(&self) -> Vec<f32> {
        self.data.read().unwrap().clone()
    }

    pub fn write_slice(&self, values: &[f32]) {
        let mut lock = self.data.write().unwrap();
        let len = values.len().min(lock.len());
        lock[..len].copy_from_slice(&values[..len]);
    }
}
