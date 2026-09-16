//! # tricore
//!
//! Pure Rust, host-agnostic WGPU graphics runtime with first-class offscreen rendering,
//! 2D batching, and 3D scene core.
//!
//! `tricore` provides the reusable graphics engine foundations for:
//! - Dioxus applications via the `trioxus` facade
//! - Embedded host plugins such as SketchUp C SDK overlays
//! - Standalone and headless simulation, test, and render pipelines

pub mod batch2d;
pub mod camera;
pub mod handles;
pub mod math;
pub mod readback;
pub mod renderer;
pub mod scene3d;
pub mod targets;

// Re-export glam for convenient math access across ecosystem
pub use glam;

// Flat re-exports for high-frequency ergonomics
pub use batch2d::*;
pub use camera::*;
pub use handles::*;
pub use math::*;
pub use readback::*;
pub use renderer::*;
pub use scene3d::*;
pub use targets::*;

/// Crate version string.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_exists() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_cube_geometry() {
        let cube = Mesh3D::unit_cube();
        assert_eq!(cube.vertices.len(), 24);
        assert_eq!(cube.indices.len(), 36);
    }
}
