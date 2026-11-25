use glam::{Mat4, Vec3};
use tricore::{
    wgpu, ChannelSwizzle, CompositeFinalizePipeline, DepthReadbackOptions, DepthReadbackSurface,
    OccluderDrawGroup, OccluderInstance, OccluderVertex3D, OffscreenTarget, OutputAlphaMode,
    Preview3dPassData, Preview3dPipeline, PreviewVertex3D, ReadbackOptions, ReadbackSurface,
    SceneOccluderBatch, SourceAlphaMode, TargetDimensions,
};

use crate::shared_gpu;

fn fullscreen_quad(z: f32, color: [f32; 4]) -> ([PreviewVertex3D; 4], [u32; 6]) {
    let verts = [
        PreviewVertex3D::new([-1.0, -1.0, z], color),
        PreviewVertex3D::new([1.0, -1.0, z], color),
        PreviewVertex3D::new([1.0, 1.0, z], color),
        PreviewVertex3D::new([-1.0, 1.0, z], color),
    ];
    let indices = [0, 1, 2, 0, 2, 3];
    (verts, indices)
}

fn render_and_readback_straight(
    w: u32,
    h: u32,
    preview: &mut Preview3dPipeline,
    view_proj: Mat4,
    data: &Preview3dPassData<'_>,
    occluders: Option<&SceneOccluderBatch<'_>>,
) -> (Vec<u8>, Vec<f32>) {
    let gpu = shared_gpu();
    let dims = TargetDimensions::new(w, h, 1.0);
    let main_target = OffscreenTarget::new(
        &gpu.device,
        dims,
        wgpu::TextureFormat::Rgba8Unorm,
        Some(wgpu::TextureFormat::Depth32Float),
    );
    let out_target = OffscreenTarget::new(&gpu.device, dims, wgpu::TextureFormat::Rgba8Unorm, None);
    let finalize =
        CompositeFinalizePipeline::new(&gpu.device, wgpu::TextureFormat::Rgba8Unorm, None);

    let mut color_rb = ReadbackSurface::new(ReadbackOptions {
        swizzle: ChannelSwizzle::None,
        y_flip: false,
        unpremultiply: false,
    });
    color_rb.ensure_dimensions(&gpu.device, w, h);

    let mut depth_rb = DepthReadbackSurface::new(DepthReadbackOptions { y_flip: false });
    depth_rb.ensure_dimensions(&gpu.device, w, h);

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("depth_peel_test_encoder"),
        });

    preview.render_depth_tested_with_occluders(
        &gpu.device,
        &gpu.queue,
        &mut encoder,
        &main_target.color_view,
        main_target.depth_view.as_ref(),
        [w, h],
        Some(wgpu::Color::TRANSPARENT),
        Some(1.0),
        view_proj,
        data,
        occluders,
    );

    finalize.finalize_single(
        &gpu.device,
        &mut encoder,
        &out_target.color_view,
        &main_target.color_view,
        SourceAlphaMode::Premultiplied,
        OutputAlphaMode::Straight,
    );

    color_rb
        .copy_from_texture(&mut encoder, &out_target.color_texture)
        .unwrap();
    depth_rb
        .copy_from_texture(&mut encoder, main_target.depth_texture.as_ref().unwrap())
        .unwrap();

    gpu.queue.submit(std::iter::once(encoder.finish()));

    let pixels = color_rb.map_and_repack(&gpu.device).unwrap().to_vec();
    let depths = depth_rb.map_and_repack(&gpu.device).unwrap().to_vec();
    (pixels, depths)
}

/// P65-G1: Steeply sloped single k=1 translucent quad has 0 self-peeled pixels and 100% uniform alpha.
#[test]
fn p65_g1_sloped_single_layer_zero_self_peeling() {
    let gpu = shared_gpu();
    let w = 128_u32;
    let h = 128_u32;
    let mut preview = Preview3dPipeline::new(
        &gpu.device,
        wgpu::TextureFormat::Rgba8Unorm,
        Some(wgpu::TextureFormat::Depth32Float),
    );

    // Steeply sloped diagonal quad across the full viewport: z varies from 0.1 to 0.95.
    let color = [0.2, 0.8, 0.4, 0.5];
    let verts = [
        PreviewVertex3D::new([-1.0, -1.0, 0.10], color),
        PreviewVertex3D::new([1.0, -1.0, 0.65], color),
        PreviewVertex3D::new([1.0, 1.0, 0.95], color),
        PreviewVertex3D::new([-1.0, 1.0, 0.40], color),
    ];
    let idx = [0_u32, 1, 2, 0, 2, 3];

    let (pixels, depths) = render_and_readback_straight(
        w,
        h,
        &mut preview,
        Mat4::IDENTITY,
        &Preview3dPassData {
            translucent_triangles: &verts,
            translucent_indices: Some(&idx),
            ..Default::default()
        },
        None,
    );

    let first_rgba = &pixels[0..4];
    for i in 0..(w * h) as usize {
        let px = &pixels[i * 4..i * 4 + 4];
        assert_eq!(
            px, first_rgba,
            "P65-G1: sloped k=1 quad self-peeled or varied at pixel {i}: got {px:?}, expected {first_rgba:?}"
        );
        let z = depths[i];
        assert!(
            (0.09..=0.96).contains(&z),
            "P65-G1: unexpected depth z={z} at pixel {i}"
        );
    }

    let expected_r = (0.2_f32 * 255.0).round() as i32;
    let expected_g = (0.8_f32 * 255.0).round() as i32;
    let expected_b = (0.4_f32 * 255.0).round() as i32;
    let expected_a = (0.5_f32 * 255.0).round() as i32;
    assert!((first_rgba[0] as i32 - expected_r).abs() <= 2, "got {first_rgba:?}");
    assert!((first_rgba[1] as i32 - expected_g).abs() <= 2, "got {first_rgba:?}");
    assert!((first_rgba[2] as i32 - expected_b).abs() <= 2, "got {first_rgba:?}");
    assert!((first_rgba[3] as i32 - expected_a).abs() <= 2, "got {first_rgba:?}");
}

/// P65-G2: Mid-depth Pass A0 occluder between front Red (z=0.3) and back Green (z=0.7) in both
/// forward and reverse submission order produces 0% noise, 0% green bleed, and smooth Porter-Duff Over.
#[test]
fn p65_g2_mid_depth_occluder_zero_noise_and_zero_green_bleed() {
    let gpu = shared_gpu();
    let w = 64_u32;
    let h = 64_u32;
    let mut preview = Preview3dPipeline::new(
        &gpu.device,
        wgpu::TextureFormat::Rgba8Unorm,
        Some(wgpu::TextureFormat::Depth32Float),
    );

    // Pass A0 opaque scene occluder on the left half (NDC x in [-1.0, 0.0]) at z = 0.5.
    let occ_verts = [
        OccluderVertex3D { position: [-1.0, -1.0, 0.5] },
        OccluderVertex3D { position: [0.0, -1.0, 0.5] },
        OccluderVertex3D { position: [0.0, 1.0, 0.5] },
        OccluderVertex3D { position: [-1.0, 1.0, 0.5] },
    ];
    let occ_indices = [0_u32, 1, 2, 0, 2, 3];
    let occ_instances = [OccluderInstance {
        model_cols: Mat4::IDENTITY.to_cols_array_2d(),
        clip_range: [0, 0],
    }];
    let occ_groups = [OccluderDrawGroup {
        index_start: 0,
        index_count: 6,
        base_vertex: 0,
        instance_start: 0,
        instance_count: 1,
    }];
    let occluder_batch = SceneOccluderBatch {
        revision: 1,
        vertices: &occ_verts,
        indices: &occ_indices,
        instances: &occ_instances,
        draw_groups: &occ_groups,
        clip_planes: &[],
    };

    let (green_v, _) = fullscreen_quad(0.7, [0.0, 1.0, 0.0, 0.5]);
    let (red_v, _) = fullscreen_quad(0.3, [1.0, 0.0, 0.0, 0.5]);
    let indices = [0_u32, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7];

    // Forward order: Green then Red.
    let mut fwd_verts = Vec::new();
    fwd_verts.extend_from_slice(&green_v);
    fwd_verts.extend_from_slice(&red_v);
    let (fwd_px, fwd_z) = render_and_readback_straight(
        w,
        h,
        &mut preview,
        Mat4::IDENTITY,
        &Preview3dPassData {
            translucent_triangles: &fwd_verts,
            translucent_indices: Some(&indices),
            ..Default::default()
        },
        Some(&occluder_batch),
    );

    // Reverse order: Red then Green.
    let mut rev_verts = Vec::new();
    rev_verts.extend_from_slice(&red_v);
    rev_verts.extend_from_slice(&green_v);
    let (rev_px, rev_z) = render_and_readback_straight(
        w,
        h,
        &mut preview,
        Mat4::IDENTITY,
        &Preview3dPassData {
            translucent_triangles: &rev_verts,
            translucent_indices: Some(&indices),
            ..Default::default()
        },
        Some(&occluder_batch),
    );

    assert_eq!(fwd_px, rev_px, "P65-G2: OIT color must be identical in forward and reverse order");
    assert_eq!(fwd_z, rev_z, "P65-G2: OIT depth must be identical in forward and reverse order");

    // Left half (over Pass A0 occluder): 100% uniform smooth Red (straight = (255, 0, 0, 128)), zero green bleed.
    let ref_occ = &fwd_px[(16 * w + 8) as usize * 4..(16 * w + 8) as usize * 4 + 4];
    assert!(ref_occ[0] >= 253 && ref_occ[1] == 0 && ref_occ[2] == 0, "over occluder must be pure Red, got {ref_occ:?}");
    assert!((ref_occ[3] as i32 - 128).abs() <= 2, "over occluder alpha must be ~0.5 (128), got {ref_occ:?}");

    for y in 0..h {
        for x in 0..(w / 2 - 1) {
            let i = (y * w + x) as usize;
            let px = &fwd_px[i * 4..i * 4 + 4];
            assert_eq!(px, ref_occ, "P65-G2: non-zero variance over occluder at ({x},{y})");
            assert!((fwd_z[i] - 0.3).abs() < 1e-4, "P65-G2: expected frag_depth ≈ 0.3 at ({x},{y}), got {}", fwd_z[i]);
        }
    }

    // Right half (outside occluder): 100% uniform smooth Porter-Duff Over:
    // premul = (0.5, 0.25, 0.0, 0.75) -> straight = (2/3, 1/3, 0.0, 0.75) -> ~(170, 85, 0, 191).
    let ref_out = &fwd_px[(16 * w + 48) as usize * 4..(16 * w + 48) as usize * 4 + 4];
    assert!((ref_out[0] as i32 - 170).abs() <= 2, "outside occluder R mismatch: {ref_out:?}");
    assert!((ref_out[1] as i32 - 85).abs() <= 2, "outside occluder G mismatch: {ref_out:?}");
    assert_eq!(ref_out[2], 0, "outside occluder B mismatch: {ref_out:?}");
    assert!((ref_out[3] as i32 - 191).abs() <= 2, "outside occluder A mismatch: {ref_out:?}");

    for y in 0..h {
        for x in (w / 2 + 1)..w {
            let i = (y * w + x) as usize;
            let px = &fwd_px[i * 4..i * 4 + 4];
            assert_eq!(px, ref_out, "P65-G2: non-zero variance outside occluder at ({x},{y})");
            assert!((fwd_z[i] - 0.3).abs() < 1e-4, "P65-G2: expected frag_depth ≈ 0.3 at ({x},{y}), got {}", fwd_z[i]);
        }
    }
}

/// P65-G3: Coplanar preview quad on Pass A0 wall survives (DepthBias), while preview quad 1 inch behind wall is clipped.
#[test]
fn p65_g3_coplanar_preview_survives_and_behind_wall_clipped() {
    let gpu = shared_gpu();
    let w = 64_u32;
    let h = 64_u32;
    let mut preview = Preview3dPipeline::new(
        &gpu.device,
        wgpu::TextureFormat::Rgba8Unorm,
        Some(wgpu::TextureFormat::Depth32Float),
    );

    // Perspective camera (SketchUp inches): eye at (0, 0, 200) looking at (0, 0, 0), near = 1 in, far = 1000 in.
    let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 200.0), Vec3::ZERO, Vec3::Y);
    let proj = Mat4::perspective_rh(45.0_f32.to_radians(), 1.0, 1.0, 1000.0);
    let view_proj = proj * view;

    // Pass A0 wall covering full viewport at z_world = 0.0 (200 in from camera).
    let occ_verts = [
        OccluderVertex3D { position: [-100.0, -100.0, 0.0] },
        OccluderVertex3D { position: [100.0, -100.0, 0.0] },
        OccluderVertex3D { position: [100.0, 100.0, 0.0] },
        OccluderVertex3D { position: [-100.0, 100.0, 0.0] },
    ];
    let occ_indices = [0_u32, 1, 2, 0, 2, 3];
    let occ_instances = [OccluderInstance {
        model_cols: Mat4::IDENTITY.to_cols_array_2d(),
        clip_range: [0, 0],
    }];
    let occ_groups = [OccluderDrawGroup {
        index_start: 0,
        index_count: 6,
        base_vertex: 0,
        instance_start: 0,
        instance_count: 1,
    }];
    let occluder_batch = SceneOccluderBatch {
        revision: 1,
        vertices: &occ_verts,
        indices: &occ_indices,
        instances: &occ_instances,
        draw_groups: &occ_groups,
        clip_planes: &[],
    };

    // Left half: coplanar translucent Blue quad at z_world = 0.0 (must survive via DepthBias).
    // Right half: behind-wall translucent Red quad at z_world = -1.0 (1 in behind wall, must be clipped).
    let verts = [
        PreviewVertex3D::new([-80.0, -80.0, 0.0], [0.0, 0.0, 1.0, 0.5]),
        PreviewVertex3D::new([-5.0, -80.0, 0.0], [0.0, 0.0, 1.0, 0.5]),
        PreviewVertex3D::new([-5.0, 80.0, 0.0], [0.0, 0.0, 1.0, 0.5]),
        PreviewVertex3D::new([-80.0, 80.0, 0.0], [0.0, 0.0, 1.0, 0.5]),
        PreviewVertex3D::new([5.0, -80.0, -1.0], [1.0, 0.0, 0.0, 0.5]),
        PreviewVertex3D::new([80.0, -80.0, -1.0], [1.0, 0.0, 0.0, 0.5]),
        PreviewVertex3D::new([80.0, 80.0, -1.0], [1.0, 0.0, 0.0, 0.5]),
        PreviewVertex3D::new([5.0, 80.0, -1.0], [1.0, 0.0, 0.0, 0.5]),
    ];
    let indices = [0_u32, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7];

    let (pixels, depths) = render_and_readback_straight(
        w,
        h,
        &mut preview,
        view_proj,
        &Preview3dPassData {
            translucent_triangles: &verts,
            translucent_indices: Some(&indices),
            ..Default::default()
        },
        Some(&occluder_batch),
    );

    // Sample left half (coplanar quad): must survive as Blue with alpha ~0.5.
    let left_idx = (32 * w + 16) as usize;
    let left_px = &pixels[left_idx * 4..left_idx * 4 + 4];
    assert!(
        left_px[0] == 0 && left_px[1] == 0 && left_px[2] >= 253 && (left_px[3] as i32 - 128).abs() <= 2,
        "P65-G3: coplanar preview quad must survive Pass A0 depth bias, got {left_px:?}"
    );
    assert!(depths[left_idx] < 1.0, "P65-G3: coplanar preview must write depth < 1.0");

    // Sample right half (1 in behind wall): must be completely clipped.
    let right_idx = (32 * w + 48) as usize;
    let right_px = &pixels[right_idx * 4..right_idx * 4 + 4];
    assert_eq!(
        right_px,
        &[0, 0, 0, 0],
        "P65-G3: preview quad 1 in behind wall must be clipped, got {right_px:?}"
    );
    assert_eq!(depths[right_idx], 1.0, "P65-G3: clipped pixel must have depth == 1.0");
}

/// P65-G4: Per-group section plane clipping (`plane = [0, 0, -1, 10]`), unclipped sibling instance,
/// and mirrored instance (`det(M) < 0`).
#[test]
fn p65_g4_per_group_section_plane_and_mirrored_instance() {
    let gpu = shared_gpu();
    let w = 96_u32;
    let h = 64_u32;
    let mut preview = Preview3dPipeline::new(
        &gpu.device,
        wgpu::TextureFormat::Rgba8Unorm,
        Some(wgpu::TextureFormat::Depth32Float),
    );

    // Orthographic camera looking along +Y at the XZ plane:
    // world x in [-15, 15], world z in [0, 20] (z = 10 is vertical center; top half is z > 10, bottom is z < 10).
    let view = Mat4::look_at_rh(Vec3::new(0.0, -20.0, 10.0), Vec3::new(0.0, 0.0, 10.0), Vec3::Z);
    let proj = Mat4::orthographic_rh(-15.0, 15.0, -10.0, 10.0, 1.0, 50.0);
    let view_proj = proj * view;

    // Shared unit column mesh in local space: x in [-4, 4], y = 0, z in [0, 20].
    let occ_verts = [
        OccluderVertex3D { position: [-4.0, 0.0, 0.0] },
        OccluderVertex3D { position: [4.0, 0.0, 0.0] },
        OccluderVertex3D { position: [4.0, 0.0, 20.0] },
        OccluderVertex3D { position: [-4.0, 0.0, 20.0] },
    ];
    let occ_indices = [0_u32, 1, 2, 0, 2, 3];

    // 3 instances across the viewport:
    // - Left (x = -10): clipped by plane [0, 0, -1, 10] (keeps z <= 10, discards z > 10).
    // - Center (x = 0): mirrored (det(M) = -1 < 0) and clipped by plane [0, 0, -1, 10].
    // - Right (x = +10): unclipped (clip_range = [0, 0]).
    let m_left = Mat4::from_translation(Vec3::new(-10.0, 0.0, 0.0));
    let m_center_mirrored = Mat4::from_scale(Vec3::new(-1.0, 1.0, 1.0));
    assert!(m_center_mirrored.determinant() < 0.0);
    let m_right = Mat4::from_translation(Vec3::new(10.0, 0.0, 0.0));

    let occ_instances = [
        OccluderInstance {
            model_cols: m_left.to_cols_array_2d(),
            clip_range: [0, 1],
        },
        OccluderInstance {
            model_cols: m_center_mirrored.to_cols_array_2d(),
            clip_range: [0, 1],
        },
        OccluderInstance {
            model_cols: m_right.to_cols_array_2d(),
            clip_range: [0, 0],
        },
    ];
    let occ_groups = [OccluderDrawGroup {
        index_start: 0,
        index_count: 6,
        base_vertex: 0,
        instance_start: 0,
        instance_count: 3,
    }];
    let clip_planes = [[0.0_f32, 0.0, -1.0, 10.0]];
    let occluder_batch = SceneOccluderBatch {
        revision: 1,
        vertices: &occ_verts,
        indices: &occ_indices,
        instances: &occ_instances,
        draw_groups: &occ_groups,
        clip_planes: &clip_planes,
    };

    // Translucent Green backdrop behind all 3 occluders at y = 5.0.
    let verts = [
        PreviewVertex3D::new([-15.0, 5.0, 0.0], [0.0, 1.0, 0.0, 0.5]),
        PreviewVertex3D::new([15.0, 5.0, 0.0], [0.0, 1.0, 0.0, 0.5]),
        PreviewVertex3D::new([15.0, 5.0, 20.0], [0.0, 1.0, 0.0, 0.5]),
        PreviewVertex3D::new([-15.0, 5.0, 20.0], [0.0, 1.0, 0.0, 0.5]),
    ];
    let indices = [0_u32, 1, 2, 0, 2, 3];

    let (pixels, _depths) = render_and_readback_straight(
        w,
        h,
        &mut preview,
        view_proj,
        &Preview3dPassData {
            translucent_triangles: &verts,
            translucent_indices: Some(&indices),
            ..Default::default()
        },
        Some(&occluder_batch),
    );

    // Pixel row y = 16 is top half (z ≈ 15 > 10); row y = 48 is bottom half (z ≈ 5 < 10).
    // Columns: x = 16 (left), x = 48 (center mirrored), x = 80 (right unclipped).
    let px_at = |x: u32, y: u32| -> [u8; 4] {
        let i = (y * w + x) as usize;
        [pixels[i * 4], pixels[i * 4 + 1], pixels[i * 4 + 2], pixels[i * 4 + 3]]
    };

    // Left (clipped): top visible (Green, a ≈ 128), bottom occluded (transparent).
    let left_top = px_at(16, 16);
    assert!(left_top[1] >= 253 && (left_top[3] as i32 - 128).abs() <= 2, "P65-G4: left top should be visible Green, got {left_top:?}");
    assert_eq!(px_at(16, 48), [0, 0, 0, 0], "P65-G4: left bottom should be occluded");

    // Center (mirrored det(M) < 0, clipped): top visible (Green, a ≈ 128), bottom occluded.
    let center_top = px_at(48, 16);
    assert!(center_top[1] >= 253 && (center_top[3] as i32 - 128).abs() <= 2, "P65-G4: mirrored center top should be visible Green, got {center_top:?}");
    assert_eq!(px_at(48, 48), [0, 0, 0, 0], "P65-G4: mirrored center bottom should be occluded");

    // Right (unclipped): both top and bottom occluded.
    assert_eq!(px_at(80, 16), [0, 0, 0, 0], "P65-G4: unclipped right top should be occluded");
    assert_eq!(px_at(80, 48), [0, 0, 0, 0], "P65-G4: unclipped right bottom should be occluded");
}

/// P65-G5: Opaque 1px line clipped behind Pass A0 occluder and 100% solid in front;
/// k = 5 tail overflow at far perspective depth z in [0.996, 0.9995] blends smoothly without FLOAT32_BLENDABLE.
#[test]
fn p65_g5_opaque_line_clipping_and_k5_tail_smoothness() {
    let gpu = shared_gpu();
    let w = 64_u32;
    let h = 64_u32;
    let mut preview = Preview3dPipeline::new(
        &gpu.device,
        wgpu::TextureFormat::Rgba8Unorm,
        Some(wgpu::TextureFormat::Depth32Float),
    );

    // Part A: Pass A0 occluder at z = 0.5 covering full screen.
    // Front opaque yellow line at z = 0.2 (y = 0.25 -> row ~24) must be 100% solid.
    // Back opaque cyan line at z = 0.8 (y = -0.25 -> row ~40) must be completely clipped.
    // Translucent backdrop at z = 0.9 triggers the depth-peel + Pass A0 pipeline.
    let occ_verts = [
        OccluderVertex3D { position: [-1.0, -1.0, 0.5] },
        OccluderVertex3D { position: [1.0, -1.0, 0.5] },
        OccluderVertex3D { position: [1.0, 1.0, 0.5] },
        OccluderVertex3D { position: [-1.0, 1.0, 0.5] },
    ];
    let occ_indices = [0_u32, 1, 2, 0, 2, 3];
    let occ_instances = [OccluderInstance {
        model_cols: Mat4::IDENTITY.to_cols_array_2d(),
        clip_range: [0, 0],
    }];
    let occ_groups = [OccluderDrawGroup {
        index_start: 0,
        index_count: 6,
        base_vertex: 0,
        instance_start: 0,
        instance_count: 1,
    }];
    let occluder_batch = SceneOccluderBatch {
        revision: 1,
        vertices: &occ_verts,
        indices: &occ_indices,
        instances: &occ_instances,
        draw_groups: &occ_groups,
        clip_planes: &[],
    };

    let (quad_v, quad_i) = fullscreen_quad(0.9, [0.0, 0.0, 1.0, 0.5]);
    let lines = [
        // Front yellow line (z = 0.2, in front of occluder at 0.5)
        PreviewVertex3D::new([-1.0, 0.25, 0.2], [1.0, 1.0, 0.0, 1.0]),
        PreviewVertex3D::new([1.0, 0.25, 0.2], [1.0, 1.0, 0.0, 1.0]),
        // Back cyan line (z = 0.8, behind occluder at 0.5)
        PreviewVertex3D::new([-1.0, -0.25, 0.8], [0.0, 1.0, 1.0, 1.0]),
        PreviewVertex3D::new([1.0, -0.25, 0.8], [0.0, 1.0, 1.0, 1.0]),
    ];

    let (pixels, depths) = render_and_readback_straight(
        w,
        h,
        &mut preview,
        Mat4::IDENTITY,
        &Preview3dPassData {
            lines: &lines,
            translucent_triangles: &quad_v,
            translucent_indices: Some(&quad_i),
            ..Default::default()
        },
        Some(&occluder_batch),
    );

    // Verify front yellow line is 100% solid across an entire row, and zero cyan pixels exist anywhere.
    let mut front_line_pixels = 0usize;
    for y in 0..h {
        let mut row_yellow = 0usize;
        for x in 0..w {
            let i = (y * w + x) as usize;
            let px = &pixels[i * 4..i * 4 + 4];
            if (depths[i] - 0.2).abs() < 1e-3 {
                assert_eq!(px, &[255, 255, 0, 255], "P65-G5: front line pixel ({x},{y}) must be solid yellow");
                row_yellow += 1;
            } else {
                // Everything else (including back cyan line at z=0.8 and blue quad at z=0.9) is occluded by Pass A0.
                assert_eq!(px, &[0, 0, 0, 0], "P65-G5: occluded pixel ({x},{y}) must be transparent, got {px:?}");
            }
        }
        if row_yellow > 0 {
            assert_eq!(row_yellow, w as usize, "P65-G5: front line row {y} must have zero dotting");
            front_line_pixels += row_yellow;
        }
    }
    assert!(front_line_pixels >= w as usize, "P65-G5: expected at least one full solid front line row");

    // Part B: k = 5 overlapping translucent quads at far perspective depth z in [0.996, 0.9995].
    let layers = [
        (0.9960_f32, [1.0_f32, 0.0, 0.0, 0.5]), // Peel 0: Red    (w = 0.5)
        (0.9970_f32, [0.0_f32, 1.0, 0.0, 0.5]), // Peel 1: Green  (w = 0.25)
        (0.9980_f32, [0.0_f32, 0.0, 1.0, 0.5]), // Peel 2: Blue   (w = 0.125)
        (0.9988_f32, [1.0_f32, 1.0, 0.0, 0.5]), // Peel 3: Yellow (w = 0.0625)
        (0.9995_f32, [0.0_f32, 1.0, 1.0, 0.5]), // Tail (5th): Cyan (w = 0.03125)
    ];
    let mut verts = Vec::new();
    let mut indices = Vec::new();
    for (i, &(z, color)) in layers.iter().enumerate() {
        let (v, _) = fullscreen_quad(z, color);
        let base = (i * 4) as u32;
        verts.extend_from_slice(&v);
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    let (k5_pixels, k5_depths) = render_and_readback_straight(
        w,
        h,
        &mut preview,
        Mat4::IDENTITY,
        &Preview3dPassData {
            translucent_triangles: &verts,
            translucent_indices: Some(&indices),
            ..Default::default()
        },
        None,
    );

    // Exact Porter-Duff Over:
    // premul = (0.5 + 0.0625, 0.25 + 0.0625 + 0.03125, 0.125 + 0.03125, 0.96875)
    //        = (0.5625, 0.34375, 0.15625, 0.96875)
    // straight = premul.rgb / 0.96875 = (18/31, 11/31, 5/31) ≈ (148, 90, 41), alpha ≈ 247.
    let exp_r = ((0.5625_f64 / 0.96875) * 255.0).round() as i32;
    let exp_g = ((0.34375_f64 / 0.96875) * 255.0).round() as i32;
    let exp_b = ((0.15625_f64 / 0.96875) * 255.0).round() as i32;
    let exp_a = (0.96875_f64 * 255.0).round() as i32;

    let ref_px = &k5_pixels[0..4];
    assert!((ref_px[0] as i32 - exp_r).abs() <= 3, "P65-G5 k=5 R mismatch: got {ref_px:?}, exp ({exp_r},{exp_g},{exp_b},{exp_a})");
    assert!((ref_px[1] as i32 - exp_g).abs() <= 3, "P65-G5 k=5 G mismatch: got {ref_px:?}, exp ({exp_r},{exp_g},{exp_b},{exp_a})");
    assert!((ref_px[2] as i32 - exp_b).abs() <= 3, "P65-G5 k=5 B mismatch: got {ref_px:?}, exp ({exp_r},{exp_g},{exp_b},{exp_a})");
    assert!((ref_px[3] as i32 - exp_a).abs() <= 3, "P65-G5 k=5 A mismatch: got {ref_px:?}, exp ({exp_r},{exp_g},{exp_b},{exp_a})");

    for i in 0..(w * h) as usize {
        assert_eq!(&k5_pixels[i * 4..i * 4 + 4], ref_px, "P65-G5: k=5 smooth OIT must have 0 variance at pixel {i}");
        assert!((k5_depths[i] - 0.9960).abs() < 1e-4, "P65-G5: k=5 first_z must be ≈ 0.9960, got {}", k5_depths[i]);
    }
}

/// P65-G6: Revision-keyed GPU buffer reuse (0 re-uploads on same revision) and zero-translucent fast-path
/// allocates no scratch and performs 0 occluder uploads.
#[test]
fn p65_g6_revision_buffer_reuse_and_zero_translucent_fast_path() {
    let gpu = shared_gpu();
    let w = 32_u32;
    let h = 32_u32;
    let mut preview = Preview3dPipeline::new(
        &gpu.device,
        wgpu::TextureFormat::Rgba8Unorm,
        Some(wgpu::TextureFormat::Depth32Float),
    );
    assert!(!preview.has_depth_peel_scratch());
    assert_eq!(preview.occluder_upload_count(), 0);

    let occ_verts = [
        OccluderVertex3D { position: [-1.0, -1.0, 0.5] },
        OccluderVertex3D { position: [1.0, -1.0, 0.5] },
        OccluderVertex3D { position: [1.0, 1.0, 0.5] },
        OccluderVertex3D { position: [-1.0, 1.0, 0.5] },
    ];
    let occ_indices = [0_u32, 1, 2, 0, 2, 3];
    let occ_instances = [OccluderInstance {
        model_cols: Mat4::IDENTITY.to_cols_array_2d(),
        clip_range: [0, 0],
    }];
    let occ_groups = [OccluderDrawGroup {
        index_start: 0,
        index_count: 6,
        base_vertex: 0,
        instance_start: 0,
        instance_count: 1,
    }];
    let mut batch = SceneOccluderBatch {
        revision: 1,
        vertices: &occ_verts,
        indices: &occ_indices,
        instances: &occ_instances,
        draw_groups: &occ_groups,
        clip_planes: &[],
    };

    // 1) Zero-translucent fast-path: even when occluders is Some, scratch is NOT allocated and 0 uploads occur.
    let (opaque_v, opaque_i) = fullscreen_quad(0.4, [0.0, 1.0, 0.0, 1.0]);
    let (pixels, depths) = render_and_readback_straight(
        w,
        h,
        &mut preview,
        Mat4::IDENTITY,
        &Preview3dPassData {
            opaque_triangles: &opaque_v,
            opaque_indices: Some(&opaque_i),
            ..Default::default()
        },
        Some(&batch),
    );
    assert!(!preview.has_depth_peel_scratch(), "P65-G6: zero-translucent fast path must not allocate scratch");
    assert_eq!(preview.occluder_upload_count(), 0, "P65-G6: zero-translucent fast path must not upload occluders");
    let center = (16 * w + 16) as usize;
    assert_eq!(&pixels[center * 4..center * 4 + 4], &[0, 255, 0, 255]);
    assert!((depths[center] - 0.4).abs() < 1e-4);

    // 2) Translucent render with revision = 1 -> uploads once.
    let (trans_v, trans_i) = fullscreen_quad(0.3, [1.0, 0.0, 0.0, 0.5]);
    let trans_data = Preview3dPassData {
        translucent_triangles: &trans_v,
        translucent_indices: Some(&trans_i),
        ..Default::default()
    };
    let _ = render_and_readback_straight(w, h, &mut preview, Mat4::IDENTITY, &trans_data, Some(&batch));
    assert!(preview.has_depth_peel_scratch());
    assert_eq!(preview.occluder_upload_count(), 1);

    // 3) Repeated render with same revision = 1 -> 0 re-uploads.
    let _ = render_and_readback_straight(w, h, &mut preview, Mat4::IDENTITY, &trans_data, Some(&batch));
    assert_eq!(preview.occluder_upload_count(), 1);

    // 4) Revision bump to 2 -> uploads again (count == 2).
    batch.revision = 2;
    let _ = render_and_readback_straight(w, h, &mut preview, Mat4::IDENTITY, &trans_data, Some(&batch));
    assert_eq!(preview.occluder_upload_count(), 2);

    // 5) clear_occluder_cache forces re-upload even on same revision = 2 (count == 3).
    preview.clear_occluder_cache();
    let _ = render_and_readback_straight(w, h, &mut preview, Mat4::IDENTITY, &trans_data, Some(&batch));
    assert_eq!(preview.occluder_upload_count(), 3);
}
