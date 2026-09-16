//! 3D scene foundations, vertex definitions, and mesh structures.

use glam::{Quat, Vec3};

/// Standard 3D vertex with position, normal, and texture coordinates.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex3D {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

impl Vertex3D {
    pub fn new(position: [f32; 3], normal: [f32; 3], uv: [f32; 2]) -> Self {
        Self { position, normal, uv }
    }
}

/// 3D transform representation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform3D {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Default for Transform3D {
    fn default() -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

impl Transform3D {
    pub fn matrix(&self) -> glam::Mat4 {
        glam::Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }
}

/// Basic PBR material properties.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Material3D {
    pub base_color: [f32; 4],
    pub roughness: f32,
    pub metallic: f32,
}

impl Default for Material3D {
    fn default() -> Self {
        Self {
            base_color: [1.0, 1.0, 1.0, 1.0],
            roughness: 0.5,
            metallic: 0.0,
        }
    }
}

/// Simple 3D indexed mesh.
pub struct Mesh3D {
    pub vertices: Vec<Vertex3D>,
    pub indices: Vec<u32>,
}

impl Mesh3D {
    pub fn new(vertices: Vec<Vertex3D>, indices: Vec<u32>) -> Self {
        Self { vertices, indices }
    }

    /// Creates a unit cube mesh centered at origin.
    pub fn unit_cube() -> Self {
        let p = 0.5;
        let n = -0.5;
        let vertices = vec![
            // Front face
            Vertex3D::new([n, n, p], [0.0, 0.0, 1.0], [0.0, 1.0]),
            Vertex3D::new([p, n, p], [0.0, 0.0, 1.0], [1.0, 1.0]),
            Vertex3D::new([p, p, p], [0.0, 0.0, 1.0], [1.0, 0.0]),
            Vertex3D::new([n, p, p], [0.0, 0.0, 1.0], [0.0, 0.0]),
            // Back face
            Vertex3D::new([p, n, n], [0.0, 0.0, -1.0], [0.0, 1.0]),
            Vertex3D::new([n, n, n], [0.0, 0.0, -1.0], [1.0, 1.0]),
            Vertex3D::new([n, p, n], [0.0, 0.0, -1.0], [1.0, 0.0]),
            Vertex3D::new([p, p, n], [0.0, 0.0, -1.0], [0.0, 0.0]),
            // Top face
            Vertex3D::new([n, p, p], [0.0, 1.0, 0.0], [0.0, 1.0]),
            Vertex3D::new([p, p, p], [0.0, 1.0, 0.0], [1.0, 1.0]),
            Vertex3D::new([p, p, n], [0.0, 1.0, 0.0], [1.0, 0.0]),
            Vertex3D::new([n, p, n], [0.0, 1.0, 0.0], [0.0, 0.0]),
            // Bottom face
            Vertex3D::new([n, n, n], [0.0, -1.0, 0.0], [0.0, 1.0]),
            Vertex3D::new([p, n, n], [0.0, -1.0, 0.0], [1.0, 1.0]),
            Vertex3D::new([p, n, p], [0.0, -1.0, 0.0], [1.0, 0.0]),
            Vertex3D::new([n, n, p], [0.0, -1.0, 0.0], [0.0, 0.0]),
            // Right face
            Vertex3D::new([p, n, p], [1.0, 0.0, 0.0], [0.0, 1.0]),
            Vertex3D::new([p, n, n], [1.0, 0.0, 0.0], [1.0, 1.0]),
            Vertex3D::new([p, p, n], [1.0, 0.0, 0.0], [1.0, 0.0]),
            Vertex3D::new([p, p, p], [1.0, 0.0, 0.0], [0.0, 0.0]),
            // Left face
            Vertex3D::new([n, n, n], [-1.0, 0.0, 0.0], [0.0, 1.0]),
            Vertex3D::new([n, n, p], [-1.0, 0.0, 0.0], [1.0, 1.0]),
            Vertex3D::new([n, p, p], [-1.0, 0.0, 0.0], [1.0, 0.0]),
            Vertex3D::new([n, p, n], [-1.0, 0.0, 0.0], [0.0, 0.0]),
        ];

        let mut indices = Vec::with_capacity(36);
        for f in 0..6 {
            let offset = f * 4;
            indices.extend_from_slice(&[
                offset,
                offset + 1,
                offset + 2,
                offset,
                offset + 2,
                offset + 3,
            ]);
        }

        Self { vertices, indices }
    }
}
