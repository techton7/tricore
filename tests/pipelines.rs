use tricore::glam::Mat4;
use tricore::{
    AlphaBlitPipeline, OffscreenTarget, Preview3dPassData, Preview3dPipeline,
    PreviewVertex3D, ReadbackOptions, ReadbackSurface, ScreenTriangleBatch,
    ScreenTrianglePipeline, SourceAlphaMode, TargetDimensions, ViewportRect,
};

fn pixel_at(pixels: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
    let idx = ((y * width + x) * 4) as usize;
    [pixels[idx], pixels[idx + 1], pixels[idx + 2], pixels[idx + 3]]
}

#[test]
fn preview3d_depth_occlusion_lines_points_and_translucency() {
    let Some(gpu) = crate::try_shared_gpu() else {
        return;
    };

    let size = 32_u32;
    let target = OffscreenTarget::new(
        &gpu.device,
        TargetDimensions::new(size, size, 1.0),
        wgpu::TextureFormat::Rgba8Unorm,
        Some(wgpu::TextureFormat::Depth32Float),
    );
    let mut preview = Preview3dPipeline::new(
        &gpu.device,
        wgpu::TextureFormat::Rgba8Unorm,
        Some(wgpu::TextureFormat::Depth32Float),
    );

    let proj = Mat4::orthographic_rh(-1.0, 1.0, -1.0, 1.0, 0.0, 10.0);

    // Near red quad (z=-1.0) covering [-0.5, 0.5]^2, then far green quad (z=-5.0) covering full [-1, 1]^2.
    // Because far green is drawn AFTER near red, depth testing must keep center red!
    let opaque_verts = [
        // Near red quad (0..4)
        PreviewVertex3D::new([-0.5, -0.5, -1.0], [1.0, 0.0, 0.0, 1.0]),
        PreviewVertex3D::new([0.5, -0.5, -1.0], [1.0, 0.0, 0.0, 1.0]),
        PreviewVertex3D::new([0.5, 0.5, -1.0], [1.0, 0.0, 0.0, 1.0]),
        PreviewVertex3D::new([-0.5, 0.5, -1.0], [1.0, 0.0, 0.0, 1.0]),
        // Far green quad (4..8)
        PreviewVertex3D::new([-1.0, -1.0, -5.0], [0.0, 1.0, 0.0, 1.0]),
        PreviewVertex3D::new([1.0, -1.0, -5.0], [0.0, 1.0, 0.0, 1.0]),
        PreviewVertex3D::new([1.0, 1.0, -5.0], [0.0, 1.0, 0.0, 1.0]),
        PreviewVertex3D::new([-1.0, 1.0, -5.0], [0.0, 1.0, 0.0, 1.0]),
    ];
    let opaque_indices = [
        0, 1, 2, 0, 2, 3, // near red
        4, 5, 6, 4, 6, 7, // far green
    ];

    // Translucent blue quad at z=-0.5 covering [-0.25, 0.25]^2 (alpha 0.5) over near red.
    let translucent_verts = [
        PreviewVertex3D::new([-0.25, -0.25, -0.5], [0.0, 0.0, 1.0, 0.5]),
        PreviewVertex3D::new([0.25, -0.25, -0.5], [0.0, 0.0, 1.0, 0.5]),
        PreviewVertex3D::new([0.25, 0.25, -0.5], [0.0, 0.0, 1.0, 0.5]),
        PreviewVertex3D::new([-0.25, 0.25, -0.5], [0.0, 0.0, 1.0, 0.5]),
    ];
    let translucent_indices = [0, 1, 2, 0, 2, 3];

    // White line across top edge (z=-0.2) and yellow point at bottom-right (0.875, -0.875, -0.2)
    let lines = [
        PreviewVertex3D::new([-0.9, 0.85, -0.2], [1.0, 1.0, 1.0, 1.0]),
        PreviewVertex3D::new([0.9, 0.85, -0.2], [1.0, 1.0, 1.0, 1.0]),
    ];
    let strip = [
        PreviewVertex3D::new([-0.9, -0.85, -0.2], [1.0, 1.0, 0.0, 1.0]),
        PreviewVertex3D::new([0.0, -0.85, -0.2], [1.0, 1.0, 0.0, 1.0]),
    ];
    let points = [PreviewVertex3D::new([0.875, -0.875, -0.2], [0.0, 1.0, 1.0, 1.0])];

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    preview.render(
        &gpu.device,
        &gpu.queue,
        &mut encoder,
        &target.color_view,
        target.depth_view.as_ref(),
        Some(wgpu::Color::TRANSPARENT),
        Some(1.0),
        proj,
        &Preview3dPassData {
            points: &points,
            lines: &lines,
            line_strips: &[&strip],
            opaque_triangles: &opaque_verts,
            opaque_indices: Some(&opaque_indices),
            translucent_triangles: &translucent_verts,
            translucent_indices: Some(&translucent_indices),
        },
    );
    gpu.queue.submit(std::iter::once(encoder.finish()));

    let mut rb = ReadbackSurface::new(ReadbackOptions::default());
    let pixels = rb
        .readback_sync(&gpu.device, &gpu.queue, &target.color_texture)
        .expect("preview3d readback");

    // Outer area (2, 16) -> far green
    assert_eq!(pixel_at(pixels, size, 2, 16), [0, 255, 0, 255]);
    // Near red only area (10, 10) -> red (not overwritten by far green)
    assert_eq!(pixel_at(pixels, size, 10, 10), [255, 0, 0, 255]);
    // Center (16, 16) -> translucent blue (0.5) over red -> (~127, 0, ~128, 255)
    let center = pixel_at(pixels, size, 16, 16);
    assert!(
        (center[0] as i32 - 127).abs() <= 2 && center[1] == 0 && (center[2] as i32 - 128).abs() <= 2,
        "unexpected center blend pixel: {center:?}"
    );
    // White line at y_ndc = 0.85 (row 2)
    assert!(
        (1..4).any(|y| pixel_at(pixels, size, 16, y) == [255, 255, 255, 255]),
        "expected white line pixel near (16, 2)"
    );
    // Yellow line strip at y_ndc = -0.85 (row 29)
    assert!(
        (28..31).any(|y| pixel_at(pixels, size, 8, y) == [255, 255, 0, 255]),
        "expected yellow line_strip pixel near (8, 29)"
    );
    // Cyan point at (0.875, -0.875) -> near (30, 30)
    assert!(
        (29..31).any(|y| (29..31).any(|x| pixel_at(pixels, size, x, y) == [0, 255, 255, 255])),
        "expected cyan point pixel near (30, 30)"
    );
}

#[test]
fn gizmo2d_top_left_pixel_coords_premul_and_depth_always() {
    let Some(gpu) = crate::try_shared_gpu() else {
        return;
    };

    let size = 64_u32;
    let target = OffscreenTarget::new(
        &gpu.device,
        TargetDimensions::new(size, size, 1.0),
        wgpu::TextureFormat::Rgba8Unorm,
        Some(wgpu::TextureFormat::Depth32Float),
    );

    let mut preview = Preview3dPipeline::new(
        &gpu.device,
        wgpu::TextureFormat::Rgba8Unorm,
        Some(wgpu::TextureFormat::Depth32Float),
    );
    let mut gizmo = ScreenTrianglePipeline::new(
        &gpu.device,
        wgpu::TextureFormat::Rgba8Unorm,
        Some(wgpu::TextureFormat::Depth32Float),
    );

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    // Clear color to opaque black and depth to 0.0 (nearest possible depth!) so only CompareFunction::Always passes.
    preview.render(
        &gpu.device,
        &gpu.queue,
        &mut encoder,
        &target.color_view,
        target.depth_view.as_ref(),
        Some(wgpu::Color::BLACK),
        Some(0.0),
        Mat4::IDENTITY,
        &Preview3dPassData::default(),
    );

    // Draw top-left rect (10, 20)..(30, 40) with linear premultiplied 50% alpha red [0.5, 0.0, 0.0, 0.5].
    let verts = [[10.0, 20.0], [30.0, 20.0], [30.0, 40.0], [10.0, 40.0]];
    let colors = [[0.5, 0.0, 0.0, 0.5]; 4];
    let indices = [0, 1, 2, 0, 2, 3];

    gizmo.render(
        &gpu.device,
        &gpu.queue,
        &mut encoder,
        &target.color_view,
        target.depth_view.as_ref(),
        ViewportRect::new(0.0, 0.0, size as f32, size as f32),
        &ScreenTriangleBatch {
            vertices: &verts,
            colors: &colors,
            indices: &indices,
        },
    );
    gpu.queue.submit(std::iter::once(encoder.finish()));

    let mut rb = ReadbackSurface::new(ReadbackOptions::default());
    let pixels = rb
        .readback_sync(&gpu.device, &gpu.queue, &target.color_texture)
        .expect("gizmo2d readback");

    // Inside top-left rect (20, 30): straight linear red = 1.0 -> sRGB 1.0 -> premul by 0.5 -> ~128 over black
    let inside = pixel_at(pixels, size, 20, 30);
    assert!(
        (inside[0] as i32 - 128).abs() <= 2 && inside[1] == 0 && inside[2] == 0 && inside[3] == 255,
        "inside gizmo rect pixel={inside:?}"
    );
    // Outside rect (5, 5) and mirrored Y position (20, 50) remain black.
    assert_eq!(pixel_at(pixels, size, 5, 5), [0, 0, 0, 255]);
    assert_eq!(pixel_at(pixels, size, 20, 50), [0, 0, 0, 255]);
}

#[test]
fn alpha_blit_composites_straight_and_premul_with_load_op_load() {
    let Some(gpu) = crate::try_shared_gpu() else {
        return;
    };

    let size = 16_u32;
    let dst = OffscreenTarget::new(
        &gpu.device,
        TargetDimensions::new(size, size, 1.0),
        wgpu::TextureFormat::Rgba8Unorm,
        None,
    );
    let src_tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("alpha_blit_test_src"),
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let src_view = src_tex.create_view(&wgpu::TextureViewDescriptor::default());

    // Left half (x < 8): straight semi-transparent red [255, 0, 0, 128]; right half: transparent [0, 0, 0, 0].
    let mut src_data = vec![0u8; (size * size * 4) as usize];
    for y in 0..size {
        for x in 0..8 {
            let idx = ((y * size + x) * 4) as usize;
            src_data[idx..idx + 4].copy_from_slice(&[255, 0, 0, 128]);
        }
    }
    gpu.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &src_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &src_data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(size * 4),
            rows_per_image: Some(size),
        },
        wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
    );

    let blit = AlphaBlitPipeline::new(&gpu.device, wgpu::TextureFormat::Rgba8Unorm);
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let _clear = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &dst.color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLUE),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
    }
    blit.composite(
        &gpu.device,
        &mut encoder,
        &src_view,
        &dst.color_view,
        SourceAlphaMode::Straight,
    );
    gpu.queue.submit(std::iter::once(encoder.finish()));

    let mut rb = ReadbackSurface::new(ReadbackOptions::default());
    let pixels = rb
        .readback_sync(&gpu.device, &gpu.queue, &dst.color_texture)
        .expect("alpha_blit straight readback");

    let left = pixel_at(pixels, size, 2, 8);
    assert!(
        (left[0] as i32 - 128).abs() <= 2 && left[1] == 0 && (left[2] as i32 - 127).abs() <= 2 && left[3] == 255,
        "unexpected straight composite left pixel: {left:?}"
    );
    // Right half must preserve existing blue target via LoadOp::Load.
    assert_eq!(pixel_at(pixels, size, 12, 8), [0, 0, 255, 255]);

    // Also test Premultiplied mode: [128, 0, 0, 128] over clear blue -> same ~[128, 0, 127, 255].
    for y in 0..size {
        for x in 0..8 {
            let idx = ((y * size + x) * 4) as usize;
            src_data[idx..idx + 4].copy_from_slice(&[128, 0, 0, 128]);
        }
    }
    gpu.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &src_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &src_data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(size * 4),
            rows_per_image: Some(size),
        },
        wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
    );

    let mut encoder2 = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let _clear = encoder2.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &dst.color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLUE),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
    }
    blit.composite(
        &gpu.device,
        &mut encoder2,
        &src_view,
        &dst.color_view,
        SourceAlphaMode::Premultiplied,
    );
    gpu.queue.submit(std::iter::once(encoder2.finish()));

    let pixels2 = rb
        .readback_sync(&gpu.device, &gpu.queue, &dst.color_texture)
        .expect("alpha_blit premul readback");
    let left2 = pixel_at(pixels2, size, 2, 8);
    assert!(
        (left2[0] as i32 - 128).abs() <= 2 && left2[1] == 0 && (left2[2] as i32 - 127).abs() <= 2 && left2[3] == 255,
        "unexpected premul composite left pixel: {left2:?}"
    );
    assert_eq!(pixel_at(pixels2, size, 12, 8), [0, 0, 255, 255]);
}
