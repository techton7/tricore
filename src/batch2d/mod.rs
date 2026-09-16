//! 2D high-volume batch rendering foundations (lines, cubic beziers, grids, and proxy quads).

/// GPU instanced line segment.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LineInstance {
    pub start: [f32; 2],
    pub end: [f32; 2],
    pub width: f32,
    pub _padding: f32,
    pub color: [f32; 4],
}

impl LineInstance {
    pub fn new(start: [f32; 2], end: [f32; 2], width: f32, color: [f32; 4]) -> Self {
        Self {
            start,
            end,
            width,
            _padding: 0.0,
            color,
        }
    }
}

/// GPU instanced cubic Bezier curve for node graphs and flow connections.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CubicBezierInstance {
    pub p0: [f32; 2],
    pub p1: [f32; 2],
    pub p2: [f32; 2],
    pub p3: [f32; 2],
    pub stroke_width: f32,
    pub _padding: f32,
    pub color: [f32; 4],
}

impl CubicBezierInstance {
    pub fn new(
        p0: [f32; 2],
        p1: [f32; 2],
        p2: [f32; 2],
        p3: [f32; 2],
        stroke_width: f32,
        color: [f32; 4],
    ) -> Self {
        Self {
            p0,
            p1,
            p2,
            p3,
            stroke_width,
            _padding: 0.0,
            color,
        }
    }
}

/// GPU instanced 2D quad for zoomed-out node LOD proxy rendering.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct QuadInstance {
    pub min: [f32; 2],
    pub max: [f32; 2],
    pub color: [f32; 4],
    pub corner_radius: f32,
    pub _padding: [f32; 3],
}

impl QuadInstance {
    pub fn new(min: [f32; 2], max: [f32; 2], color: [f32; 4], corner_radius: f32) -> Self {
        Self {
            min,
            max,
            color,
            corner_radius,
            _padding: [0.0; 3],
        }
    }
}

/// Uniform configuration for background grid rendering.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GridConfig {
    pub cell_size: f32,
    pub line_width: f32,
    pub _padding: [f32; 2],
    pub color: [f32; 4],
}

impl Default for GridConfig {
    fn default() -> Self {
        Self {
            cell_size: 20.0,
            line_width: 1.0,
            _padding: [0.0; 2],
            color: [0.3, 0.3, 0.3, 1.0],
        }
    }
}
