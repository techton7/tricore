use tricore::glam::Mat4;
use tricore::{
    ChannelSwizzle, DepthReadbackOptions, DepthReadbackSurface, OffscreenTarget,
    OffscreenTargetOptions, Preview3dPassData, Preview3dPipeline, PreviewVertex3D, ReadbackOptions,
    ReadbackSurface, ScreenTriangleBatch, ScreenTrianglePipeline, TargetDimensions, ViewportRect,
};

#[test]
fn offscreen_target_storage_binding_and_ensure_dimensions() {
    let Some(gpu) = crate::try_shared_gpu() else {
        return;
    };

    let mut target = OffscreenTarget::new(
        &gpu.device,
        TargetDimensions::new(16, 16, 1.0),
        wgpu::TextureFormat::Rgba8Unorm,
        Some(wgpu::TextureFormat::Depth32Float),
    );
    assert!(target.color_usage.contains(wgpu::TextureUsages::STORAGE_BINDING));
    assert_eq!(target.color_texture.width(), 16);
    assert_eq!(target.color_texture.height(), 16);
    assert!(target.depth_texture.is_some());

    // Equivalent physical dimensions (8 * 2.0 == 16) should not reallocate.
    assert!(!target.ensure_dimensions(&gpu.device, TargetDimensions::new(8, 8, 2.0)));
    assert_eq!(target.color_texture.width(), 16);

    // Changed physical dimensions must reallocate while preserving STORAGE_BINDING and depth format.
    assert!(target.ensure_dimensions(&gpu.device, TargetDimensions::new(24, 18, 1.0)));
    assert_eq!(target.color_texture.width(), 24);
    assert_eq!(target.color_texture.height(), 18);
    assert!(target.color_usage.contains(wgpu::TextureUsages::STORAGE_BINDING));
    assert_eq!(target.depth_format, Some(wgpu::TextureFormat::Depth32Float));
    assert_eq!(
        target.depth_texture.as_ref().map(|t| (t.width(), t.height())),
        Some((24, 18))
    );

    let custom = OffscreenTarget::new_with_usage(
        &gpu.device,
        TargetDimensions::new(8, 8, 1.0),
        wgpu::TextureFormat::Rgba8Unorm,
        None,
        OffscreenTargetOptions::storage_compatible().color_usage,
    );
    assert!(custom.color_usage.contains(wgpu::TextureUsages::STORAGE_BINDING));
    assert!(custom.depth_texture.is_none());
}

#[test]
fn readback_surface_unpads_non_256_width_swizzles_flips_and_caches() {
    let Some(gpu) = crate::try_shared_gpu() else {
        return;
    };

    // 13x7 => 52 bytes/row unpadded, requiring 256-byte padded staging stride.
    let w = 13_u32;
    let h = 7_u32;
    let target = OffscreenTarget::new(
        &gpu.device,
        TargetDimensions::new(w, h, 1.0),
        wgpu::TextureFormat::Rgba8Unorm,
        None,
    );

    // Clear to opaque blue, then paint top-left (0..6, 0..3) opaque linear red via ScreenTrianglePipeline.
    let mut gizmo = ScreenTrianglePipeline::new(&gpu.device, wgpu::TextureFormat::Rgba8Unorm, None);
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let _clear = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target.color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 1.0,
                        a: 1.0,
                    }),
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
    let verts = [[0.0, 0.0], [6.0, 0.0], [6.0, 3.0], [0.0, 3.0]];
    let colors = [[1.0, 0.0, 0.0, 1.0]; 4];
    let indices = [0, 1, 2, 0, 2, 3];
    gizmo.render(
        &gpu.device,
        &gpu.queue,
        &mut encoder,
        &target.color_view,
        None,
        ViewportRect::new(0.0, 0.0, w as f32, h as f32),
        &ScreenTriangleBatch {
            vertices: &verts,
            colors: &colors,
            indices: &indices,
        },
    );
    gpu.queue.submit(std::iter::once(encoder.finish()));

    let mut rgba_rb = ReadbackSurface::new(ReadbackOptions {
        swizzle: ChannelSwizzle::None,
        y_flip: false,
        unpremultiply: false,
    });
    assert!(rgba_rb.ensure_dimensions(&gpu.device, w, h));
    assert!(!rgba_rb.ensure_dimensions(&gpu.device, w, h));

    let pixels = rgba_rb
        .readback_sync(&gpu.device, &gpu.queue, &target.color_texture)
        .expect("readback_sync rgba")
        .to_vec();
    assert_eq!(pixels.len(), (w * h * 4) as usize);
    assert_eq!(rgba_rb.width(), w);
    assert_eq!(rgba_rb.height(), h);
    assert_eq!(rgba_rb.unpadded_row_pitch(), w * 4);
    // Top-left (1, 1) -> red
    let tl = ((1 * w + 1) * 4) as usize;
    assert_eq!(&pixels[tl..tl + 4], &[255, 0, 0, 255]);
    // Bottom-right (12, 6) -> blue
    let br = ((6 * w + 12) * 4) as usize;
    assert_eq!(&pixels[br..br + 4], &[0, 0, 255, 255]);
    // Persistent cache stays intact for idle frames.
    assert_eq!(rgba_rb.cached_pixels(), pixels.as_slice());

    // SwapRB + y_flip test
    let mut bgra_flip = ReadbackSurface::new(ReadbackOptions {
        swizzle: ChannelSwizzle::SwapRB,
        y_flip: true,
        unpremultiply: false,
    });
    let flipped = bgra_flip
        .readback_sync(&gpu.device, &gpu.queue, &target.color_texture)
        .expect("readback_sync SwapRB y_flip");
    // Original row 1 (red at x=1) moves to row (7 - 1 - 1) = 5, swizzled to BGRA [0, 0, 255, 255]
    let flipped_red = ((5 * w + 1) * 4) as usize;
    assert_eq!(&flipped[flipped_red..flipped_red + 4], &[0, 0, 255, 255]);
    // Original row 6 (blue at x=12) moves to row 0, swizzled to BGRA [255, 0, 0, 255]
    let flipped_blue = (12 * 4) as usize;
    assert_eq!(&flipped[flipped_blue..flipped_blue + 4], &[255, 0, 0, 255]);
}

#[test]
fn depth_readback_surface_unpads_non_256_width_flips_and_caches_bytes() {
    let Some(gpu) = crate::try_shared_gpu() else {
        return;
    };

    // 13x7 => 52 bytes/row unpadded, requiring 256-byte padded staging stride.
    let w = 13_u32;
    let h = 7_u32;
    let target = OffscreenTarget::new(
        &gpu.device,
        TargetDimensions::new(w, h, 1.0),
        wgpu::TextureFormat::Rgba8Unorm,
        Some(wgpu::TextureFormat::Depth32Float),
    );
    let mut preview = Preview3dPipeline::new(
        &gpu.device,
        wgpu::TextureFormat::Rgba8Unorm,
        Some(wgpu::TextureFormat::Depth32Float),
    );

    // Orthographic proj: z = -2.5 in [0, -10] maps to NDC z = 0.25.
    let proj = Mat4::orthographic_rh(-1.0, 1.0, -1.0, 1.0, 0.0, 10.0);
    // Top half of screen (NDC y in [0, 1], i.e. rows 0..3) covered by quad at z = -2.5.
    let verts = [
        PreviewVertex3D::new([-1.0, 0.0, -2.5], [1.0, 0.0, 0.0, 1.0]),
        PreviewVertex3D::new([1.0, 0.0, -2.5], [1.0, 0.0, 0.0, 1.0]),
        PreviewVertex3D::new([1.0, 1.0, -2.5], [1.0, 0.0, 0.0, 1.0]),
        PreviewVertex3D::new([-1.0, 1.0, -2.5], [1.0, 0.0, 0.0, 1.0]),
    ];
    let indices = [0, 1, 2, 0, 2, 3];

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
            opaque_triangles: &verts,
            opaque_indices: Some(&indices),
            ..Default::default()
        },
    );
    gpu.queue.submit(std::iter::once(encoder.finish()));

    let depth_tex = target.depth_texture.as_ref().expect("depth texture");
    let mut depth_rb = DepthReadbackSurface::new(DepthReadbackOptions { y_flip: false });
    assert!(depth_rb.ensure_dimensions(&gpu.device, w, h));
    assert!(!depth_rb.ensure_dimensions(&gpu.device, w, h));

    let depths = depth_rb
        .readback_sync(&gpu.device, &gpu.queue, depth_tex)
        .expect("depth readback")
        .to_vec();
    assert_eq!(depths.len(), (w * h) as usize);
    assert_eq!(depth_rb.width(), w);
    assert_eq!(depth_rb.height(), h);
    assert_eq!(depth_rb.cached_depths(), depths.as_slice());
    let expected_bytes: Vec<u8> = depths.iter().flat_map(|d| d.to_le_bytes()).collect();
    assert_eq!(depth_rb.cached_bytes(), expected_bytes.as_slice());

    // Top row (y=1, x=6) -> 0.25, bottom row (y=6, x=6) -> background clear 1.0.
    let top_d = depths[(1 * w + 6) as usize];
    let bot_d = depths[(6 * w + 6) as usize];
    assert!((top_d - 0.25).abs() < 1e-4, "expected ~0.25, got {top_d}");
    assert!((bot_d - 1.0).abs() < 1e-6, "expected 1.0, got {bot_d}");

    // Verify y_flip moves top row (y=1) to y=(7-1-1)=5 and bottom row (y=6) to y=0.
    depth_rb.set_options(DepthReadbackOptions { y_flip: true });
    let flipped = depth_rb
        .readback_sync(&gpu.device, &gpu.queue, depth_tex)
        .expect("depth readback y_flip");
    assert!((flipped[(5 * w + 6) as usize] - 0.25).abs() < 1e-4);
    assert!((flipped[6] - 1.0).abs() < 1e-6);
}
