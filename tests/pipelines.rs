use tricore::glam::Mat4;
use tricore::{
    AlphaBlitPipeline, ChannelSwizzle, CompositeFinalizeError, CompositeFinalizePipeline,
    DepthReadbackOptions, DepthReadbackSurface, OffscreenTarget, OutputAlphaMode,
    Preview3dPassData, Preview3dPipeline, PreviewVertex3D, ReadbackOptions, ReadbackSurface,
    ScreenTriangleBatch, ScreenTrianglePipeline, SourceAlphaMode, TargetDimensions, ViewportRect,
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

fn upload_rgba_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    size: u32,
    data: &[u8],
    label: &str,
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
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
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        data,
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
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

#[test]
fn composite_finalize_2_input_and_1_input_rgba_and_bgra() {
    let Some(gpu) = crate::try_shared_gpu() else {
        return;
    };

    let size = 8_u32;
    // Base (Premultiplied): left half (x < 4) = [0, 0, 128, 128] (50% blue premul),
    // right half (x >= 4) = [0, 0, 0, 0] (transparent, dirty test covered via single pass).
    let mut base_data = vec![0u8; (size * size * 4) as usize];
    for y in 0..size {
        for x in 0..4 {
            let idx = ((y * size + x) * 4) as usize;
            base_data[idx..idx + 4].copy_from_slice(&[0, 0, 128, 128]);
        }
    }
    let (_base_tex, base_view) =
        upload_rgba_texture(&gpu.device, &gpu.queue, size, &base_data, "cf_base");

    // Overlay (Straight):
    // top-left (x < 4, y < 4): [255, 0, 0, 128] (50% straight red over 50% premul blue)
    //   => over_pm = (0.502, 0, 0, 0.502), base_pm = (0, 0, 0.502, 0.502)
    //   => pm = (0.502, 0, 0.250, 0.752) ~ [128, 0, 64, 192]
    //   => straight = (0.667, 0, 0.332, 0.752) ~ [170, 0, 85, 192]
    // bottom-left (x < 4, y >= 4): [0, 0, 0, 0] (transparent -> keeps base [0, 0, 128, 128] / straight [0, 0, 255, 128])
    // top-right (x >= 4, y < 4): [0, 255, 0, 128] (50% straight green over transparent -> premul [0, 128, 0, 128], straight [0, 255, 0, 128])
    // bottom-right (x >= 4, y >= 4): [0, 0, 0, 0] (both transparent -> [0, 0, 0, 0])
    let mut overlay_data = vec![0u8; (size * size * 4) as usize];
    for y in 0..4 {
        for x in 0..size {
            let idx = ((y * size + x) * 4) as usize;
            if x < 4 {
                overlay_data[idx..idx + 4].copy_from_slice(&[255, 0, 0, 128]);
            } else {
                overlay_data[idx..idx + 4].copy_from_slice(&[0, 255, 0, 128]);
            }
        }
    }
    let (_over_tex, over_view) =
        upload_rgba_texture(&gpu.device, &gpu.queue, size, &overlay_data, "cf_overlay");

    let mut rb = ReadbackSurface::new(ReadbackOptions {
        swizzle: ChannelSwizzle::None,
        y_flip: false,
        unpremultiply: false,
    });

    for (dst_format, is_bgra) in [
        (wgpu::TextureFormat::Rgba8Unorm, false),
        (wgpu::TextureFormat::Bgra8Unorm, true),
    ] {
        let dst = OffscreenTarget::new(
            &gpu.device,
            TargetDimensions::new(size, size, 1.0),
            dst_format,
            None,
        );
        let pipeline = CompositeFinalizePipeline::new(&gpu.device, dst_format, None);

        // 1) 2-input Straight output
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        pipeline.composite(
            &gpu.device,
            &mut encoder,
            &dst.color_view,
            &base_view,
            &over_view,
            SourceAlphaMode::Premultiplied,
            SourceAlphaMode::Straight,
            OutputAlphaMode::Straight,
        );
        gpu.queue.submit(std::iter::once(encoder.finish()));

        let pixels = rb
            .readback_sync(&gpu.device, &gpu.queue, &dst.color_texture)
            .expect("2-input straight readback");
        let tl = pixel_at(pixels, size, 1, 1);
        let bl = pixel_at(pixels, size, 1, 6);
        let tr = pixel_at(pixels, size, 6, 1);
        let br = pixel_at(pixels, size, 6, 6);
        let (exp_tl, exp_bl, exp_tr) = if is_bgra {
            ([85_i32, 0, 170, 192], [255_u8, 0, 0, 128], [0_u8, 255, 0, 128])
        } else {
            ([170_i32, 0, 85, 192], [0_u8, 0, 255, 128], [0_u8, 255, 0, 128])
        };
        assert!(
            (tl[0] as i32 - exp_tl[0]).abs() <= 3
                && tl[1] == 0
                && (tl[2] as i32 - exp_tl[2]).abs() <= 3
                && (tl[3] as i32 - exp_tl[3]).abs() <= 2,
            "2-input straight tl ({dst_format:?}): {tl:?}"
        );
        assert_eq!(bl, exp_bl, "2-input straight bl ({dst_format:?})");
        assert_eq!(tr, exp_tr, "2-input straight tr ({dst_format:?})");
        assert_eq!(br, [0, 0, 0, 0], "2-input straight br ({dst_format:?})");

        // 2) 2-input Premultiplied output
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        pipeline.composite(
            &gpu.device,
            &mut encoder,
            &dst.color_view,
            &base_view,
            &over_view,
            SourceAlphaMode::Premultiplied,
            SourceAlphaMode::Straight,
            OutputAlphaMode::Premultiplied,
        );
        gpu.queue.submit(std::iter::once(encoder.finish()));

        let pixels = rb
            .readback_sync(&gpu.device, &gpu.queue, &dst.color_texture)
            .expect("2-input premul readback");
        let tl = pixel_at(pixels, size, 1, 1);
        let bl = pixel_at(pixels, size, 1, 6);
        let tr = pixel_at(pixels, size, 6, 1);
        let br = pixel_at(pixels, size, 6, 6);
        let (exp_tl, exp_bl, exp_tr) = if is_bgra {
            ([64_i32, 0, 128, 192], [128_u8, 0, 0, 128], [0_u8, 128, 0, 128])
        } else {
            ([128_i32, 0, 64, 192], [0_u8, 0, 128, 128], [0_u8, 128, 0, 128])
        };
        assert!(
            (tl[0] as i32 - exp_tl[0]).abs() <= 2
                && tl[1] == 0
                && (tl[2] as i32 - exp_tl[2]).abs() <= 2
                && (tl[3] as i32 - exp_tl[3]).abs() <= 2,
            "2-input premul tl ({dst_format:?}): {tl:?}"
        );
        assert_eq!(bl, exp_bl, "2-input premul bl ({dst_format:?})");
        assert_eq!(tr, exp_tr, "2-input premul tr ({dst_format:?})");
        assert_eq!(br, [0, 0, 0, 0], "2-input premul br ({dst_format:?})");

        // 3) 1-input finalize: premul [128, 0, 0, 128] -> Straight [255, 0, 0, 128] (or BGRA [0, 0, 255, 128])
        // and dirty-zero-alpha [99, 88, 77, 0] -> [0, 0, 0, 0].
        let mut single_src_data = vec![0u8; (size * size * 4) as usize];
        for y in 0..size {
            for x in 0..size {
                let idx = ((y * size + x) * 4) as usize;
                if x < 4 {
                    single_src_data[idx..idx + 4].copy_from_slice(&[128, 0, 0, 128]);
                } else {
                    single_src_data[idx..idx + 4].copy_from_slice(&[99, 88, 77, 0]);
                }
            }
        }
        let (_single_tex, single_view) = upload_rgba_texture(
            &gpu.device,
            &gpu.queue,
            size,
            &single_src_data,
            "cf_single_src",
        );

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        pipeline.finalize_single(
            &gpu.device,
            &mut encoder,
            &dst.color_view,
            &single_view,
            SourceAlphaMode::Premultiplied,
            OutputAlphaMode::Straight,
        );
        gpu.queue.submit(std::iter::once(encoder.finish()));

        let pixels = rb
            .readback_sync(&gpu.device, &gpu.queue, &dst.color_texture)
            .expect("1-input straight readback");
        let left = pixel_at(pixels, size, 1, 4);
        let right = pixel_at(pixels, size, 6, 4);
        let exp_left = if is_bgra { [0, 0, 255, 128] } else { [255, 0, 0, 128] };
        assert_eq!(left, exp_left, "1-input straight left ({dst_format:?})");
        assert_eq!(right, [0, 0, 0, 0], "1-input straight zero-alpha ({dst_format:?})");

        // 4) 1-input finalize: straight [255, 0, 0, 128] -> Premultiplied [128, 0, 0, 128] (or BGRA [0, 0, 128, 128])
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        pipeline.finalize_single(
            &gpu.device,
            &mut encoder,
            &dst.color_view,
            &over_view,
            SourceAlphaMode::Straight,
            OutputAlphaMode::Premultiplied,
        );
        gpu.queue.submit(std::iter::once(encoder.finish()));

        let pixels = rb
            .readback_sync(&gpu.device, &gpu.queue, &dst.color_texture)
            .expect("1-input premul readback");
        let tl = pixel_at(pixels, size, 1, 1);
        let bl = pixel_at(pixels, size, 1, 6);
        let exp_tl = if is_bgra { [0, 0, 128, 128] } else { [128, 0, 0, 128] };
        assert_eq!(tl, exp_tl, "1-input premul tl ({dst_format:?})");
        assert_eq!(bl, [0, 0, 0, 0], "1-input premul bl ({dst_format:?})");
    }
}

#[test]
fn combined_depth_occlusion_preview3d_gizmo_stamp_and_overlay_stamp() {
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
    let mut gizmo = ScreenTrianglePipeline::new(
        &gpu.device,
        wgpu::TextureFormat::Rgba8Unorm,
        Some(wgpu::TextureFormat::Depth32Float),
    );
    let no_depth_composite =
        CompositeFinalizePipeline::new(&gpu.device, wgpu::TextureFormat::Rgba8Unorm, None);
    let composite = CompositeFinalizePipeline::new(
        &gpu.device,
        wgpu::TextureFormat::Rgba8Unorm,
        Some(wgpu::TextureFormat::Depth32Float),
    );

    // Overlay texture: opaque white ONLY in top-right quadrant (x >= 16, y < 16), alpha == 0 everywhere else.
    let mut overlay_data = vec![0u8; (size * size * 4) as usize];
    for y in 0..16 {
        for x in 16..size {
            let idx = ((y * size + x) * 4) as usize;
            overlay_data[idx..idx + 4].copy_from_slice(&[255, 255, 255, 255]);
        }
    }
    let (_overlay_tex, overlay_view) = upload_rgba_texture(
        &gpu.device,
        &gpu.queue,
        size,
        &overlay_data,
        "depth_stamp_overlay",
    );

    // Verify error when created without depth_format.
    let mut dummy_encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    let err = no_depth_composite
        .stamp_overlay_depth(
            &gpu.device,
            &mut dummy_encoder,
            target.depth_view.as_ref().unwrap(),
            &overlay_view,
        )
        .unwrap_err();
    assert_eq!(err, CompositeFinalizeError::DepthPipelineNotConfigured);
    drop(dummy_encoder);

    // 1. Preview3D covers left half (NDC x in [-1, 0], i.e. x < 16) at z = -5.0 -> depth = 0.5.
    // Right half remains background depth 1.0.
    let proj = Mat4::orthographic_rh(-1.0, 1.0, -1.0, 1.0, 0.0, 10.0);
    let verts3d = [
        PreviewVertex3D::new([-1.0, -1.0, -5.0], [0.0, 1.0, 0.0, 1.0]),
        PreviewVertex3D::new([0.0, -1.0, -5.0], [0.0, 1.0, 0.0, 1.0]),
        PreviewVertex3D::new([0.0, 1.0, -5.0], [0.0, 1.0, 0.0, 1.0]),
        PreviewVertex3D::new([-1.0, 1.0, -5.0], [0.0, 1.0, 0.0, 1.0]),
    ];
    let idx3d = [0, 1, 2, 0, 2, 3];

    // 2. Gizmo with depth stamp covers bottom-left sub-box (2..10, 20..28).
    let gizmo_verts = [[2.0, 20.0], [10.0, 20.0], [10.0, 28.0], [2.0, 28.0]];
    let gizmo_colors = [[1.0, 0.0, 0.0, 1.0]; 4];
    let gizmo_idx = [0, 1, 2, 0, 2, 3];

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
            opaque_triangles: &verts3d,
            opaque_indices: Some(&idx3d),
            ..Default::default()
        },
    );
    gizmo.render_with_depth_stamp(
        &gpu.device,
        &gpu.queue,
        &mut encoder,
        &target.color_view,
        target.depth_view.as_ref(),
        ViewportRect::new(0.0, 0.0, size as f32, size as f32),
        &ScreenTriangleBatch {
            vertices: &gizmo_verts,
            colors: &gizmo_colors,
            indices: &gizmo_idx,
        },
    );
    composite
        .stamp_overlay_depth(
            &gpu.device,
            &mut encoder,
            target.depth_view.as_ref().unwrap(),
            &overlay_view,
        )
        .expect("stamp_overlay_depth");
    gpu.queue.submit(std::iter::once(encoder.finish()));

    let mut depth_rb = DepthReadbackSurface::new(DepthReadbackOptions { y_flip: false });
    let depths = depth_rb
        .readback_sync(
            &gpu.device,
            &gpu.queue,
            target.depth_texture.as_ref().unwrap(),
        )
        .expect("combined depth readback");

    let depth_at = |x: u32, y: u32| depths[(y * size + x) as usize];
    // Top-left (8, 8): 3D preview only -> z = 0.5
    assert!((depth_at(8, 8) - 0.5).abs() < 1e-4, "3d only: {}", depth_at(8, 8));
    // Bottom-left inside gizmo (6, 24): gizmo stamped -> z = 0.0
    assert!((depth_at(6, 24) - 0.0).abs() < 1e-6, "gizmo stamp: {}", depth_at(6, 24));
    // Bottom-left outside gizmo (12, 24): 3D preview preserved -> z = 0.5
    assert!((depth_at(12, 24) - 0.5).abs() < 1e-4, "3d preserved: {}", depth_at(12, 24));
    // Top-right inside overlay (24, 8): overlay stamped -> z = 0.0
    assert!((depth_at(24, 8) - 0.0).abs() < 1e-6, "overlay stamp: {}", depth_at(24, 8));
    // Bottom-right (24, 24): overlay.a == 0 -> background preserved -> z = 1.0
    assert!((depth_at(24, 24) - 1.0).abs() < 1e-6, "background: {}", depth_at(24, 24));
}
