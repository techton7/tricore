use glam::Mat4;
use tricore::{
    wgpu, ChannelSwizzle, CompositeFinalizePipeline, DepthReadbackOptions, DepthReadbackSurface,
    OffscreenTarget, OutputAlphaMode, Preview3dPassData, Preview3dPipeline, PreviewVertex3D,
    ReadbackOptions, ReadbackSurface, SourceAlphaMode, TargetDimensions,
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
    data: &Preview3dPassData<'_>,
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

    preview.render_depth_tested(
        &gpu.device,
        &gpu.queue,
        &mut encoder,
        &main_target.color_view,
        main_target.depth_view.as_ref(),
        [w, h],
        Some(wgpu::Color::TRANSPARENT),
        Some(1.0),
        Mat4::IDENTITY,
        data,
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

#[test]
fn depth_peel_k1_single_translucent_quad_uniform_no_self_peel() {
    let gpu = shared_gpu();
    let w = 64_u32;
    let h = 64_u32;
    let mut preview = Preview3dPipeline::new(
        &gpu.device,
        wgpu::TextureFormat::Rgba8Unorm,
        Some(wgpu::TextureFormat::Depth32Float),
    );

    let (verts, idx) = fullscreen_quad(0.5, [0.2, 0.8, 0.4, 0.5]);
    let (pixels, depths) = render_and_readback_straight(
        w,
        h,
        &mut preview,
        &Preview3dPassData {
            translucent_triangles: &verts,
            translucent_indices: Some(&idx),
            ..Default::default()
        },
    );

    let first_rgba = &pixels[0..4];
    for i in 0..(w * h) as usize {
        let px = &pixels[i * 4..i * 4 + 4];
        assert_eq!(
            px, first_rgba,
            "k=1 must be 100% uniform with zero stipple at pixel {i}"
        );
        let z = depths[i];
        assert!(
            (z - 0.5).abs() < 1e-4,
            "k=1 expected z ≈ 0.5 at pixel {i}, got {z}"
        );
    }

    // Verify straight color & alpha accuracy (no self-peeling darkening or alpha doubling).
    let expected_r = (0.2_f32 * 255.0).round() as i32;
    let expected_g = (0.8_f32 * 255.0).round() as i32;
    let expected_b = (0.4_f32 * 255.0).round() as i32;
    let expected_a = (0.5_f32 * 255.0).round() as i32;
    assert!((first_rgba[0] as i32 - expected_r).abs() <= 2, "got {first_rgba:?}");
    assert!((first_rgba[1] as i32 - expected_g).abs() <= 2, "got {first_rgba:?}");
    assert!((first_rgba[2] as i32 - expected_b).abs() <= 2, "got {first_rgba:?}");
    assert!((first_rgba[3] as i32 - expected_a).abs() <= 2, "got {first_rgba:?}");
}

#[test]
fn depth_peel_k2_two_layers_host_composite_mid_cylinder_and_far_bg() {
    let gpu = shared_gpu();
    let w = 64_u32;
    let h = 64_u32;
    let mut preview = Preview3dPipeline::new(
        &gpu.device,
        wgpu::TextureFormat::Rgba8Unorm,
        Some(wgpu::TextureFormat::Depth32Float),
    );

    // Back Green at z = 0.7 (a = 0.5), Front Red at z = 0.3 (a = 0.5).
    let (green_v, _) = fullscreen_quad(0.7, [0.0, 1.0, 0.0, 0.5]);
    let (red_v, _) = fullscreen_quad(0.3, [1.0, 0.0, 0.0, 0.5]);
    let mut verts = Vec::new();
    verts.extend_from_slice(&green_v);
    verts.extend_from_slice(&red_v);
    let indices = [0_u32, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7];

    let (pixels, depths) = render_and_readback_straight(
        w,
        h,
        &mut preview,
        &Preview3dPassData {
            translucent_triangles: &verts,
            translucent_indices: Some(&indices),
            ..Default::default()
        },
    );

    let total_px = (w * h) as usize;
    let mut red_count = 0usize;
    let mut green_count = 0usize;

    // Host simulation accumulators:
    // 1) Mid-depth white cylinder at z_cyl = 0.5, color = (1, 1, 1)
    let mut mid_sum = [0.0_f64; 3];
    // 2) Far blue background at z_bg = 0.99, color = (0, 0, 1)
    let mut far_sum = [0.0_f64; 3];

    for i in 0..total_px {
        let r = pixels[i * 4] as f64 / 255.0;
        let g = pixels[i * 4 + 1] as f64 / 255.0;
        let b = pixels[i * 4 + 2] as f64 / 255.0;
        let a = pixels[i * 4 + 3] as f64 / 255.0;
        let z = depths[i];

        // Total opacity must be 0.5 + 0.5 * 0.5 = 0.75 at every pixel.
        assert!(
            (a - 0.75).abs() < 0.02,
            "expected total alpha ≈ 0.75 at pixel {i}, got {a}"
        );

        // Every pixel must export ONLY pure Red at z ≈ 0.3 OR pure Green at z ≈ 0.7 (never blended!).
        if (z - 0.3).abs() < 1e-4 {
            assert!(
                r > 0.98 && g < 0.02 && b < 0.02,
                "z ≈ 0.3 pixel {i} must be pure Red, got ({r}, {g}, {b})"
            );
            red_count += 1;
        } else if (z - 0.7).abs() < 1e-4 {
            assert!(
                r < 0.02 && g > 0.98 && b < 0.02,
                "z ≈ 0.7 pixel {i} must be pure Green, got ({r}, {g}, {b})"
            );
            green_count += 1;
        } else {
            panic!("unexpected depth z = {z} at pixel {i}");
        }

        // Simulate SketchUp host composite against mid-depth white cylinder (z_cyl = 0.5).
        let mid_rgb = if z < 0.5 {
            [
                r * a + 1.0 * (1.0 - a),
                g * a + 1.0 * (1.0 - a),
                b * a + 1.0 * (1.0 - a),
            ]
        } else {
            [1.0, 1.0, 1.0]
        };
        // Zero green bleed: green must never exceed blue on any pixel against white cylinder.
        assert!(
            (mid_rgb[1] - mid_rgb[2]).abs() < 0.02,
            "green bleed detected at pixel {i}: {mid_rgb:?}"
        );
        for c in 0..3 {
            mid_sum[c] += mid_rgb[c];
        }

        // Simulate SketchUp host composite against far blue background (z_bg = 0.99).
        let far_rgb = if z < 0.99 {
            [
                r * a + 0.0 * (1.0 - a),
                g * a + 0.0 * (1.0 - a),
                b * a + 1.0 * (1.0 - a),
            ]
        } else {
            [0.0, 0.0, 1.0]
        };
        for c in 0..3 {
            far_sum[c] += far_rgb[c];
        }
    }

    assert!(red_count > 0 && green_count > 0);
    let mid_mean = [
        mid_sum[0] / total_px as f64,
        mid_sum[1] / total_px as f64,
        mid_sum[2] / total_px as f64,
    ];
    // Expected: 0.5 * Red + 0.5 * White = (1.0, 0.5, 0.5)
    assert!(
        (mid_mean[0] - 1.0).abs() < 0.02
            && (mid_mean[1] - 0.5).abs() < 0.02
            && (mid_mean[2] - 0.5).abs() < 0.02,
        "mid-cylinder mean mismatch: got {mid_mean:?}, expected (1.0, 0.5, 0.5)"
    );

    let far_mean = [
        far_sum[0] / total_px as f64,
        far_sum[1] / total_px as f64,
        far_sum[2] / total_px as f64,
    ];
    // Expected Porter-Duff Over: 0.5 * Red + 0.25 * Green + 0.25 * Blue = (0.5, 0.25, 0.25)
    assert!(
        (far_mean[0] - 0.5).abs() < 0.02
            && (far_mean[1] - 0.25).abs() < 0.02
            && (far_mean[2] - 0.25).abs() < 0.02,
        "far-background mean mismatch: got {far_mean:?}, expected (0.5, 0.25, 0.25)"
    );
}

#[test]
fn depth_peel_k2_reverse_submission_order_matches_forward() {
    let gpu = shared_gpu();
    let w = 64_u32;
    let h = 64_u32;
    let mut preview = Preview3dPipeline::new(
        &gpu.device,
        wgpu::TextureFormat::Rgba8Unorm,
        Some(wgpu::TextureFormat::Depth32Float),
    );

    let (green_v, _) = fullscreen_quad(0.7, [0.0, 1.0, 0.0, 0.5]);
    let (red_v, _) = fullscreen_quad(0.3, [1.0, 0.0, 0.0, 0.5]);
    let indices = [0_u32, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7];

    // Forward: Green then Red
    let mut fwd_verts = Vec::new();
    fwd_verts.extend_from_slice(&green_v);
    fwd_verts.extend_from_slice(&red_v);
    let (fwd_px, fwd_z) = render_and_readback_straight(
        w,
        h,
        &mut preview,
        &Preview3dPassData {
            translucent_triangles: &fwd_verts,
            translucent_indices: Some(&indices),
            ..Default::default()
        },
    );

    // Reverse: Front Red submitted BEFORE Back Green
    let mut rev_verts = Vec::new();
    rev_verts.extend_from_slice(&red_v);
    rev_verts.extend_from_slice(&green_v);
    let (rev_px, rev_z) = render_and_readback_straight(
        w,
        h,
        &mut preview,
        &Preview3dPassData {
            translucent_triangles: &rev_verts,
            translucent_indices: Some(&indices),
            ..Default::default()
        },
    );

    assert_eq!(fwd_px, rev_px, "OIT color must be identical regardless of submission order");
    assert_eq!(fwd_z, rev_z, "OIT depth must be identical regardless of submission order");
}

#[test]
fn depth_peel_k5_tail_overflow_preserves_far_perspective_depth() {
    let gpu = shared_gpu();
    let w = 64_u32;
    let h = 64_u32;
    let mut preview = Preview3dPipeline::new(
        &gpu.device,
        wgpu::TextureFormat::Rgba8Unorm,
        Some(wgpu::TextureFormat::Depth32Float),
    );

    // 5 overlapping translucent quads at far perspective depth z in [0.996, 0.9995].
    // Layers 0..3 are peeled; layer 4 (Cyan at z = 0.9995) overflows into the Rgba16Float 2-MRT Tail.
    let layers = [
        (0.9960_f32, [1.0_f32, 0.0, 0.0, 0.5]), // Peel 0: Red
        (0.9970_f32, [0.0_f32, 1.0, 0.0, 0.5]), // Peel 1: Green
        (0.9980_f32, [0.0_f32, 0.0, 1.0, 0.5]), // Peel 2: Blue
        (0.9988_f32, [1.0_f32, 1.0, 0.0, 0.5]), // Peel 3: Yellow
        (0.9995_f32, [0.0_f32, 1.0, 1.0, 0.5]), // Tail (5th): Cyan
    ];
    let mut verts = Vec::new();
    let mut indices = Vec::new();
    for (i, &(z, color)) in layers.iter().enumerate() {
        let (v, _) = fullscreen_quad(z, color);
        let base = (i * 4) as u32;
        verts.extend_from_slice(&v);
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    let (pixels, depths) = render_and_readback_straight(
        w,
        h,
        &mut preview,
        &Preview3dPassData {
            translucent_triangles: &verts,
            translucent_indices: Some(&indices),
            ..Default::default()
        },
    );

    let expected_total_alpha = 1.0 - 0.5_f64.powi(5); // 0.96875
    let mut tail_pixel_count = 0usize;

    for i in 0..(w * h) as usize {
        let r = pixels[i * 4];
        let g = pixels[i * 4 + 1];
        let b = pixels[i * 4 + 2];
        let a = pixels[i * 4 + 3] as f64 / 255.0;
        let z = depths[i];

        assert!(
            (a - expected_total_alpha).abs() < 0.02,
            "k=5 expected total alpha ≈ {expected_total_alpha:.4} at pixel {i}, got {a:.4}"
        );
        assert!(z < 1.0, "pixel {i} depth must be < 1.0, got {z}");

        // Tail layer is Cyan (r == 0, g == 255, b == 255) at z ≈ 0.9995.
        if r < 10 && g > 245 && b > 245 {
            tail_pixel_count += 1;
            assert!(
                (z - 0.9995).abs() < 1e-4 && z < 1.0,
                "tail layer depth lost precision near z=1.0: got {z} (expected ≈ 0.9995)"
            );
        }
    }

    // Weight of 5th layer is 0.5^5 = 0.03125 out of 0.96875 ≈ 3.22% of 4096 pixels ≈ 132 pixels.
    assert!(
        tail_pixel_count >= 100 && tail_pixel_count <= 165,
        "expected ~132 tail pixels out of 4096, got {tail_pixel_count}"
    );
}

#[test]
fn depth_peel_front_opaque_line_is_100_percent_solid_zero_dotting() {
    let gpu = shared_gpu();
    let w = 64_u32;
    let h = 64_u32;
    let mut preview = Preview3dPipeline::new(
        &gpu.device,
        wgpu::TextureFormat::Rgba8Unorm,
        Some(wgpu::TextureFormat::Depth32Float),
    );

    // Translucent blue quad behind at z = 0.5.
    let (quad_v, quad_i) = fullscreen_quad(0.5, [0.0, 0.0, 1.0, 0.5]);
    // Horizontal opaque yellow line in front at z = 0.2 across y = 0.0 (pixel row 32).
    let line = [
        PreviewVertex3D::new([-1.0, 0.0, 0.2], [1.0, 1.0, 0.0, 1.0]),
        PreviewVertex3D::new([1.0, 0.0, 0.2], [1.0, 1.0, 0.0, 1.0]),
    ];

    let (pixels, depths) = render_and_readback_straight(
        w,
        h,
        &mut preview,
        &Preview3dPassData {
            lines: &line,
            translucent_triangles: &quad_v,
            translucent_indices: Some(&quad_i),
            ..Default::default()
        },
    );

    // Find which row(s) the rasterizer placed the horizontal line on (row 31 or 32).
    let mut line_pixels = 0usize;
    for y in 30..=33 {
        let mut row_yellow = 0usize;
        for x in 0..w {
            let i = (y * w + x) as usize;
            let z = depths[i];
            if (z - 0.2).abs() < 1e-3 {
                row_yellow += 1;
                let px = &pixels[i * 4..i * 4 + 4];
                assert_eq!(
                    px,
                    &[255, 255, 0, 255],
                    "front opaque line pixel ({x}, {y}) must be 100% solid yellow, got {px:?}"
                );
            }
        }
        if row_yellow > 0 {
            assert_eq!(
                row_yellow, w as usize,
                "front opaque line on row {y} must have zero dotting (expected {w} solid pixels, got {row_yellow})"
            );
            line_pixels += row_yellow;
        }
    }
    assert!(
        line_pixels >= w as usize,
        "expected at least one full solid row for horizontal opaque line, got {line_pixels}"
    );
}

#[test]
fn depth_peel_zero_translucent_fast_path_allocates_no_scratch() {
    let gpu = shared_gpu();
    let w = 32_u32;
    let h = 32_u32;
    let mut preview = Preview3dPipeline::new(
        &gpu.device,
        wgpu::TextureFormat::Rgba8Unorm,
        Some(wgpu::TextureFormat::Depth32Float),
    );
    assert!(!preview.has_depth_peel_scratch());

    let (opaque_v, opaque_i) = fullscreen_quad(0.4, [0.0, 1.0, 0.0, 1.0]);
    let (pixels, depths) = render_and_readback_straight(
        w,
        h,
        &mut preview,
        &Preview3dPassData {
            opaque_triangles: &opaque_v,
            opaque_indices: Some(&opaque_i),
            ..Default::default()
        },
    );

    assert!(
        !preview.has_depth_peel_scratch(),
        "zero-translucent fast path must NOT allocate depth-peeling scratch textures"
    );

    let center = (16 * w + 16) as usize;
    assert_eq!(&pixels[center * 4..center * 4 + 4], &[0, 255, 0, 255]);
    assert!((depths[center] - 0.4).abs() < 1e-4);
}
