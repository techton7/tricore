//! Camera abstractions for 2D viewports, 3D scenes, and external host matrix injection.

use glam::{Mat4, Vec2, Vec3};

/// 2D pan/zoom viewport camera with rotation support.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera2D {
    pub pan: Vec2,
    pub zoom: f32,
    pub rotation: f32,
}

impl Default for Camera2D {
    fn default() -> Self {
        Self {
            pan: Vec2::ZERO,
            zoom: 1.0,
            rotation: 0.0,
        }
    }
}

impl Camera2D {
    pub fn new(pan: Vec2, zoom: f32) -> Self {
        Self {
            pan,
            zoom,
            rotation: 0.0,
        }
    }

    /// Computes the 2D view matrix (pan and rotation).
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(
            Vec3::new(self.zoom, self.zoom, 1.0),
            glam::Quat::from_rotation_z(self.rotation),
            Vec3::new(self.pan.x, self.pan.y, 0.0),
        )
    }

    /// Computes the orthographic projection matrix for a given viewport size.
    pub fn proj_matrix(&self, width: f32, height: f32) -> Mat4 {
        Mat4::orthographic_rh(
            -width * 0.5,
            width * 0.5,
            -height * 0.5,
            height * 0.5,
            -1000.0,
            1000.0,
        )
    }

    /// Combined view-projection matrix.
    pub fn view_proj_matrix(&self, width: f32, height: f32) -> Mat4 {
        self.proj_matrix(width, height) * self.view_matrix()
    }
}

/// 3D perspective camera for 3D viewports and orbit/flying controls.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera3D {
    pub position: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fov_y_rad: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for Camera3D {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 5.0, 10.0),
            target: Vec3::ZERO,
            up: Vec3::Y,
            fov_y_rad: 45.0_f32.to_radians(),
            near: 0.1,
            far: 1000.0,
        }
    }
}

impl Camera3D {
    pub fn new(position: Vec3, target: Vec3) -> Self {
        Self {
            position,
            target,
            ..Default::default()
        }
    }

    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.target, self.up)
    }

    pub fn proj_matrix(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_rh(self.fov_y_rad, aspect, self.near, self.far)
    }

    pub fn view_proj_matrix(&self, aspect: f32) -> Mat4 {
        self.proj_matrix(aspect) * self.view_matrix()
    }
}

/// Raw camera definition for injecting pre-computed matrices from external hosts (e.g. SketchUp C SDK).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RawCamera {
    pub view: Mat4,
    pub proj: Mat4,
}

impl RawCamera {
    pub fn new(view: Mat4, proj: Mat4) -> Self {
        Self { view, proj }
    }

    pub fn view_proj_matrix(&self) -> Mat4 {
        self.proj * self.view
    }

    pub fn inv_view_proj_matrix(&self) -> Mat4 {
        self.view_proj_matrix().inverse()
    }
}
