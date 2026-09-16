//! Viewport-space coordinate math and conversions.
//!
//! All math functions in this module are host-neutral and work in terms of normalized
//! or pixel viewport coordinates, without assuming a specific browser or OS window environment.

use glam::{Mat4, Vec2, Vec3, Vec4};

/// A 3D ray defined by an origin and a normalized direction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ray3D {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray3D {
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self {
            origin,
            direction: direction.normalize_or_zero(),
        }
    }
}

/// Host-neutral viewport rectangle in physical or logical pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewportRect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl ViewportRect {
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self { x, y, width, height }
    }

    /// Converts a pixel coordinate within this viewport into Normalized Device Coordinates (NDC).
    /// NDC X: [-1.0, 1.0], Y: [-1.0, 1.0] (with Y-up).
    pub fn pixel_to_ndc(&self, pixel_pt: Vec2) -> Vec2 {
        let rel_x = (pixel_pt.x - self.x) / self.width.max(1.0);
        let rel_y = (pixel_pt.y - self.y) / self.height.max(1.0);
        Vec2::new(rel_x * 2.0 - 1.0, 1.0 - rel_y * 2.0)
    }

    /// Converts an NDC point back into viewport pixel coordinates.
    pub fn ndc_to_pixel(&self, ndc_pt: Vec2) -> Vec2 {
        let rel_x = (ndc_pt.x + 1.0) * 0.5;
        let rel_y = (1.0 - ndc_pt.y) * 0.5;
        Vec2::new(self.x + rel_x * self.width, self.y + rel_y * self.height)
    }
}

/// Transforms a 2D viewport pixel point into 2D world coordinates.
pub fn viewport_to_world_2d(
    viewport_pt: Vec2,
    viewport: ViewportRect,
    inv_view_proj: Mat4,
) -> Vec2 {
    let ndc = viewport.pixel_to_ndc(viewport_pt);
    let world_pos = inv_view_proj.project_point3(Vec3::new(ndc.x, ndc.y, 0.0));
    Vec2::new(world_pos.x, world_pos.y)
}

/// Transforms a 2D world point into viewport pixel coordinates.
pub fn world_to_viewport_2d(
    world_pt: Vec2,
    viewport: ViewportRect,
    view_proj: Mat4,
) -> Vec2 {
    let clip = view_proj * Vec4::new(world_pt.x, world_pt.y, 0.0, 1.0);
    if clip.w.abs() < 1e-6 {
        return Vec2::ZERO;
    }
    let ndc = Vec2::new(clip.x / clip.w, clip.y / clip.w);
    viewport.ndc_to_pixel(ndc)
}

/// Unprojects a 2D viewport point into a 3D ray for object picking.
pub fn viewport_to_ray(
    viewport_pt: Vec2,
    viewport: ViewportRect,
    inv_view_proj: Mat4,
) -> Ray3D {
    let ndc = viewport.pixel_to_ndc(viewport_pt);
    // Near plane (NDC z = 0.0 in WGPU standard [0, 1] depth range)
    let p_near = inv_view_proj.project_point3(Vec3::new(ndc.x, ndc.y, 0.0));
    // Far plane (NDC z = 1.0 in WGPU)
    let p_far = inv_view_proj.project_point3(Vec3::new(ndc.x, ndc.y, 1.0));
    let dir = p_far - p_near;
    Ray3D::new(p_near, dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_viewport_ndc_roundtrip() {
        let vp = ViewportRect::new(0.0, 0.0, 800.0, 600.0);
        let pt = Vec2::new(400.0, 300.0); // Center of viewport
        let ndc = vp.pixel_to_ndc(pt);
        assert!((ndc.x - 0.0).abs() < 1e-5);
        assert!((ndc.y - 0.0).abs() < 1e-5);

        let back = vp.ndc_to_pixel(ndc);
        assert!((back.x - pt.x).abs() < 1e-5);
        assert!((back.y - pt.y).abs() < 1e-5);
    }
}
