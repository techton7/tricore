//! Camera abstractions for 2D viewports, 3D scenes, and external host matrix injection.

use glam::{DMat4, Mat2, Mat4, Vec2, Vec3};

use crate::math::{ViewportRect, viewport_to_world_2d, world_to_viewport_2d};

const MIN_CAMERA_ZOOM: f32 = 1e-4;

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

    /// Computes the 2D view matrix (pan, rotation, and uniform zoom).
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(
            Vec3::new(self.zoom, self.zoom, 1.0),
            glam::Quat::from_rotation_z(self.rotation),
            Vec3::new(self.pan.x, self.pan.y, 0.0),
        )
    }

    /// Computes the orthographic projection matrix for a given viewport size (center-origin, Y-up).
    pub fn proj_matrix(&self, width: f32, height: f32) -> Mat4 {
        let w = width.max(1.0);
        let h = height.max(1.0);
        Mat4::orthographic_rh(-w * 0.5, w * 0.5, -h * 0.5, h * 0.5, -1000.0, 1000.0)
    }

    /// Combined view-projection matrix.
    pub fn view_proj_matrix(&self, width: f32, height: f32) -> Mat4 {
        self.proj_matrix(width, height) * self.view_matrix()
    }

    /// Converts a screen pixel coordinate within `viewport` to Y-up 2D world coordinates.
    pub fn screen_to_world(&self, screen_pt: Vec2, viewport: ViewportRect) -> Vec2 {
        if self.zoom.abs() < 1e-6 {
            return Vec2::ZERO;
        }
        let inv = self
            .view_proj_matrix(viewport.width, viewport.height)
            .inverse();
        viewport_to_world_2d(screen_pt, viewport, inv)
    }

    /// Converts a Y-up 2D world point to screen pixel coordinates within `viewport`.
    pub fn world_to_screen(&self, world_pt: Vec2, viewport: ViewportRect) -> Vec2 {
        world_to_viewport_2d(
            world_pt,
            viewport,
            self.view_proj_matrix(viewport.width, viewport.height),
        )
    }

    /// Updates zoom to `new_zoom` while keeping the world point under `screen_pt` stationary.
    pub fn zoom_at(&mut self, screen_pt: Vec2, viewport: ViewportRect, new_zoom: f32) {
        let clamped_zoom = if new_zoom.is_finite() {
            new_zoom.max(MIN_CAMERA_ZOOM)
        } else {
            self.zoom.max(MIN_CAMERA_ZOOM)
        };
        let old_zoom = self.zoom;
        let world_before = self.screen_to_world(screen_pt, viewport);
        self.zoom = clamped_zoom;
        self.pan += (old_zoom - clamped_zoom) * (Mat2::from_angle(self.rotation) * world_before);
    }

    /// Multiplies zoom by `factor` while keeping the world point under `screen_pt` stationary.
    pub fn zoom_by_at(&mut self, screen_pt: Vec2, viewport: ViewportRect, factor: f32) {
        if !factor.is_finite() || factor <= 0.0 {
            return;
        }
        self.zoom_at(screen_pt, viewport, self.zoom * factor);
    }

    /// Returns the axis-aligned world-space bounding rectangle visible inside `viewport`.
    pub fn visible_world_rect(&self, viewport: ViewportRect) -> ViewportRect {
        let corners = [
            Vec2::new(viewport.x, viewport.y),
            Vec2::new(viewport.x + viewport.width, viewport.y),
            Vec2::new(viewport.x, viewport.y + viewport.height),
            Vec2::new(viewport.x + viewport.width, viewport.y + viewport.height),
        ];
        let mut min = Vec2::splat(f32::INFINITY);
        let mut max = Vec2::splat(f32::NEG_INFINITY);
        for corner in corners {
            let world = self.screen_to_world(corner, viewport);
            min = min.min(world);
            max = max.max(world);
        }
        ViewportRect::new(min.x, min.y, max.x - min.x, max.y - min.y)
    }

    /// Returns a 2D affine `[a, b, c, d, e, f]` (column-major 2x3, matching `kurbo::Affine::new`)
    /// mapping Y-up world coordinates to top-left Y-down canvas pixels of size `(width, height)`.
    pub fn to_vello_affine_ortho(&self, width: f32, height: f32) -> [f64; 6] {
        let (s, c) = (self.rotation as f64).sin_cos();
        let z = self.zoom as f64;
        [
            z * c,
            -z * s,
            -z * s,
            -z * c,
            width as f64 * 0.5 + self.pan.x as f64,
            height as f64 * 0.5 - self.pan.y as f64,
        ]
    }

    /// Returns a 2D affine `[a, b, c, d, e, f]` (matching `kurbo::Affine::new`)
    /// for Y-down world space (`p_screen = pan + zoom * R(rotation) * p_world`).
    pub fn to_vello_affine_y_down(&self) -> [f64; 6] {
        let (s, c) = (self.rotation as f64).sin_cos();
        let z = self.zoom as f64;
        [
            z * c,
            z * s,
            -z * s,
            z * c,
            self.pan.x as f64,
            self.pan.y as f64,
        ]
    }

    /// Converts a screen pixel coordinate within `viewport` to Y-down 2D world coordinates.
    pub fn screen_to_world_y_down(&self, screen_pt: Vec2, viewport: ViewportRect) -> Vec2 {
        if self.zoom.abs() < 1e-6 {
            return Vec2::ZERO;
        }
        let local = screen_pt - Vec2::new(viewport.x, viewport.y);
        Mat2::from_angle(-self.rotation) * ((local - self.pan) / self.zoom)
    }

    /// Converts a Y-down 2D world point to screen pixel coordinates within `viewport`.
    pub fn world_to_screen_y_down(&self, world_pt: Vec2, viewport: ViewportRect) -> Vec2 {
        Vec2::new(viewport.x, viewport.y)
            + self.pan
            + self.zoom * (Mat2::from_angle(self.rotation) * world_pt)
    }

    /// Multiplies zoom by `factor` while keeping `screen_pt` stationary in Y-down canvas space.
    pub fn zoom_at_y_down(&mut self, screen_pt: Vec2, factor: f32) {
        if !factor.is_finite() || factor <= 0.0 || self.zoom.abs() < 1e-6 {
            return;
        }
        let old_zoom = self.zoom;
        let new_zoom = (old_zoom * factor).max(MIN_CAMERA_ZOOM);
        let ratio = new_zoom / old_zoom;
        self.pan = screen_pt - (screen_pt - self.pan) * ratio;
        self.zoom = new_zoom;
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

fn gl_row_major_to_dmat4s(view_row_major: &[f64; 16], proj_row_major: &[f64; 16]) -> (DMat4, DMat4) {
    let mut remapped_proj = *proj_row_major;
    // OpenGL NDC Z in [-1, 1] -> WGPU NDC Z in [0, 1]: Z_wgpu = 0.5 * Z_gl + 0.5 * W_gl.
    for col in 0..4 {
        remapped_proj[8 + col] = 0.5 * remapped_proj[8 + col] + 0.5 * remapped_proj[12 + col];
    }
    let view = DMat4::from_cols_array(view_row_major).transpose();
    let proj = DMat4::from_cols_array(&remapped_proj).transpose();
    (view, proj)
}

impl RawCamera {
    pub fn new(view: Mat4, proj: Mat4) -> Self {
        Self { view, proj }
    }

    /// Converts OpenGL row-major `f64` view and projection matrices (NDC Z in `[-1, 1]`)
    /// into a column-major WGPU [`RawCamera`] (NDC Z in `[0, 1]`).
    pub fn from_gl_row_major_f64(view_row_major: &[f64; 16], proj_row_major: &[f64; 16]) -> Self {
        let (view, proj) = gl_row_major_to_dmat4s(view_row_major, proj_row_major);
        Self {
            view: view.as_mat4(),
            proj: proj.as_mat4(),
        }
    }

    /// Multiplies `proj * view` in `f64` precision before narrowing to `f32` [`Mat4`],
    /// preserving accuracy for large world coordinates far from origin.
    pub fn view_proj_gl_row_major_f64(
        view_row_major: &[f64; 16],
        proj_row_major: &[f64; 16],
    ) -> Mat4 {
        let (view, proj) = gl_row_major_to_dmat4s(view_row_major, proj_row_major);
        (proj * view).as_mat4()
    }

    pub fn view_proj_matrix(&self) -> Mat4 {
        self.proj * self.view
    }

    pub fn inv_view_proj_matrix(&self) -> Mat4 {
        self.view_proj_matrix().inverse()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec4;

    #[test]
    fn camera2d_screen_world_roundtrip_with_pan_zoom_rotation_and_offset() {
        let cam = Camera2D {
            pan: Vec2::new(42.5, -18.25),
            zoom: 2.5,
            rotation: 0.65,
        };
        let vp = ViewportRect::new(15.0, 25.0, 800.0, 600.0);
        let screen = Vec2::new(317.0, 412.0);
        let world = cam.screen_to_world(screen, vp);
        let back = cam.world_to_screen(world, vp);
        assert!((back - screen).length() < 1e-3, "back={back:?} screen={screen:?}");
    }

    #[test]
    fn camera2d_zoom_at_preserves_cursor_world_point() {
        let vp = ViewportRect::new(10.0, 20.0, 1024.0, 768.0);
        let cursor = Vec2::new(712.0, 245.0);
        for rotation in [0.0_f32, 0.78] {
            let mut cam = Camera2D {
                pan: Vec2::new(-35.0, 60.0),
                zoom: 1.5,
                rotation,
            };
            let world_before = cam.screen_to_world(cursor, vp);
            cam.zoom_by_at(cursor, vp, 2.25);
            let screen_after = cam.world_to_screen(world_before, vp);
            assert!(
                (screen_after - cursor).length() < 1e-3,
                "rotation={rotation} screen_after={screen_after:?} cursor={cursor:?}"
            );
        }
    }

    #[test]
    fn camera2d_visible_world_rect_bounds_rotated_zoomed() {
        let cam = Camera2D {
            pan: Vec2::new(40.0, -20.0),
            zoom: 2.0,
            rotation: std::f32::consts::FRAC_PI_4,
        };
        let vp = ViewportRect::new(0.0, 0.0, 400.0, 200.0);
        let rect = cam.visible_world_rect(vp);
        // Rotated 400x200 box at zoom=2 has world half-dims (100, 50); rotated 45 deg -> span = (100+50)*sqrt(2)
        let expected_span = 150.0 * std::f32::consts::SQRT_2;
        assert!((rect.width - expected_span).abs() < 1e-2);
        assert!((rect.height - expected_span).abs() < 1e-2);
    }

    #[test]
    fn camera2d_to_vello_affine_ortho_matches_world_to_screen() {
        let cam = Camera2D {
            pan: Vec2::new(33.0, -47.0),
            zoom: 1.75,
            rotation: -0.42,
        };
        let w = 640.0_f32;
        let h = 480.0_f32;
        let vp = ViewportRect::new(0.0, 0.0, w, h);
        let affine = cam.to_vello_affine_ortho(w, h);
        let world_pt = Vec2::new(125.0, -80.0);

        let affine_x = (affine[0] * world_pt.x as f64 + affine[2] * world_pt.y as f64 + affine[4]) as f32;
        let affine_y = (affine[1] * world_pt.x as f64 + affine[3] * world_pt.y as f64 + affine[5]) as f32;
        let screen_pt = cam.world_to_screen(world_pt, vp);

        assert!((affine_x - screen_pt.x).abs() < 1e-3);
        assert!((affine_y - screen_pt.y).abs() < 1e-3);
    }

    #[test]
    fn camera2d_y_down_roundtrip_affine_and_zoom_at() {
        let mut cam = Camera2D {
            pan: Vec2::new(100.0, 50.0),
            zoom: 2.0,
            rotation: 0.3,
        };
        let vp = ViewportRect::new(12.0, 8.0, 800.0, 600.0);
        let world = Vec2::new(45.0, -22.0);
        let screen = cam.world_to_screen_y_down(world, vp);
        let back = cam.screen_to_world_y_down(screen, vp);
        assert!((back - world).length() < 1e-4);

        let origin_vp = ViewportRect::new(0.0, 0.0, 800.0, 600.0);
        let affine = cam.to_vello_affine_y_down();
        let ax = (affine[0] * world.x as f64 + affine[2] * world.y as f64 + affine[4]) as f32;
        let ay = (affine[1] * world.x as f64 + affine[3] * world.y as f64 + affine[5]) as f32;
        let s0 = cam.world_to_screen_y_down(world, origin_vp);
        assert!((ax - s0.x).abs() < 1e-4);
        assert!((ay - s0.y).abs() < 1e-4);

        let cursor = Vec2::new(250.0, 175.0);
        let before = cam.screen_to_world_y_down(cursor, origin_vp);
        cam.zoom_at_y_down(cursor, 1.8);
        let after = cam.world_to_screen_y_down(before, origin_vp);
        assert!((after - cursor).length() < 1e-3);
    }

    #[test]
    fn raw_camera_gl_row_major_f64_remaps_z_and_transposes() {
        // OpenGL perspective projection (near=1.0, far=10.0, fov=90 deg, aspect=1.0) in row-major f64:
        // f = 1 / tan(45 deg) = 1.0
        // row2 = [0, 0, -(f+n)/(f-n), -2*f*n/(f-n)] = [0, 0, -11/9, -20/9]
        // row3 = [0, 0, -1, 0]
        let view_rm: [f64; 16] = [
            1.0, 0.0, 0.0, 2.0,
            0.0, 1.0, 0.0, -3.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let proj_rm: [f64; 16] = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, -11.0 / 9.0, -20.0 / 9.0,
            0.0, 0.0, -1.0, 0.0,
        ];

        let raw = RawCamera::from_gl_row_major_f64(&view_rm, &proj_rm);
        let vp = raw.view_proj_matrix();

        // Point at eye z = -1.0 (near plane): world = (-2, 3, -1, 1) -> eye = (0, 0, -1, 1) -> z_wgpu = 0.0
        let c_near = vp * Vec4::new(-2.0, 3.0, -1.0, 1.0);
        assert!((c_near.z / c_near.w - 0.0).abs() < 1e-5);

        // Point at eye z = -10.0 (far plane): world = (-2, 3, -10, 1) -> eye = (0, 0, -10, 1) -> z_wgpu = 1.0
        let c_far = vp * Vec4::new(-2.0, 3.0, -10.0, 1.0);
        assert!((c_far.z / c_far.w - 1.0).abs() < 1e-5);

        // Point where GL NDC z == 0 (eye z = -20/11): z_wgpu = 0.5
        let z_mid = -20.0_f32 / 11.0;
        let c_mid = vp * Vec4::new(-2.0, 3.0, z_mid, 1.0);
        assert!((c_mid.z / c_mid.w - 0.5).abs() < 1e-5);

        // Verify transpose of translation: world (0,0,0,1) -> eye (2, -3, 0, 1)
        let eye0 = raw.view * Vec4::new(0.0, 0.0, 0.0, 1.0);
        assert!((eye0.x - 2.0).abs() < 1e-5);
        assert!((eye0.y - (-3.0)).abs() < 1e-5);
    }

    #[test]
    fn raw_camera_view_proj_gl_row_major_f64_preserves_large_offset_precision() {
        // World origin shifted by 1e7 with non-power-of-two scale (1/3):
        // f64 multiplication before narrowing avoids f32 intermediate cancellation/drift.
        let offset = 10_000_000.0_f64;
        let view_rm: [f64; 16] = [
            1.0, 0.0, 0.0, -offset,
            0.0, 1.0, 0.0, -offset,
            0.0, 0.0, 1.0, -5.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let proj_rm: [f64; 16] = [
            1.0 / 3.0, 0.0, 0.0, 0.0,
            0.0, 1.0 / 3.0, 0.0, 0.0,
            0.0, 0.0, -2.0 / 99.0, -101.0 / 99.0,
            0.0, 0.0, 0.0, 1.0,
        ];

        let vp = RawCamera::view_proj_gl_row_major_f64(&view_rm, &proj_rm);
        let expected_tx = (-offset / 3.0) as f32;
        assert!(
            (vp.w_axis.x - expected_tx).abs() <= 0.5,
            "vp.w_axis.x={} expected={}",
            vp.w_axis.x,
            expected_tx
        );

        // Verify exact transform on power-of-two ortho scale with large offset.
        let ortho_view_rm: [f64; 16] = [
            1.0, 0.0, 0.0, -1_000_000.0,
            0.0, 1.0, 0.0, -1_000_000.0,
            0.0, 0.0, 1.0, -5.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let ortho_proj_rm: [f64; 16] = [
            0.125, 0.0, 0.0, 0.0,
            0.0, 0.125, 0.0, 0.0,
            0.0, 0.0, -2.0 / 99.0, -101.0 / 99.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let vp_ortho = RawCamera::view_proj_gl_row_major_f64(&ortho_view_rm, &ortho_proj_rm);
        let clip = vp_ortho * Vec4::new(1_000_002.0, 999_997.0, 0.0, 1.0);
        assert!((clip.x - 0.25).abs() < 1e-4, "clip.x={}", clip.x);
        assert!((clip.y - (-0.375)).abs() < 1e-4, "clip.y={}", clip.y);
    }
}
