<div align="center">
  <img src="assets/icon.svg" alt="tricore logo" width="160" height="160" />
  <h1>tricore</h1>
  <p><strong>Pure Rust, host-agnostic WGPU graphics runtime with first-class offscreen rendering, 2D batching, and 3D scene core.</strong></p>

  <p>
    <a href="https://crates.io/crates/tricore"><img src="https://img.shields.io/crates/v/tricore.svg" alt="Crates.io" /></a>
    <a href="https://docs.rs/tricore"><img src="https://docs.rs/tricore/badge.svg" alt="Docs.rs" /></a>
    <a href="LICENSE-MIT"><img src="https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg" alt="License" /></a>
  </p>
</div>

---

## Overview

`tricore` is the foundational, host-neutral graphics and viewport engine for the Rust ecosystem. It is intentionally designed without coupling to any specific UI framework or window manager, enabling identical GPU-accelerated rendering pipelines across:

- **Dioxus Web & Native Desktop** (via the [`trioxus`](https://github.com/techton7/trioxus) component facade)
- **Embedded C/C++ Host Plugins** (such as SketchUp C SDK overlays with offscreen color & depth readback)
- **Headless Simulations & Automated Test Harnesses**

---

## Core Capabilities

- **Surface-Agnostic & Offscreen-First:** First-class offscreen color (`Rgba8Unorm`) and depth (`Depth32Float`) render targets, with dedicated staging buffers for CPU readback (`ColorReadbackBuffer`, `DepthReadbackBuffer`).
- **GPU Instanced 2D Batching:** High-volume parametric cubic Bézier curves, line strips, background grids, and LOD proxy quads designed for responsive flow graphs and CAD editors.
- **Host-Neutral Viewport & Camera Math:** Unbiased projection, view matrices, and coordinate inversion (`viewport_to_world_2d`, `world_to_viewport_2d`, `viewport_to_ray`) with support for raw external host matrix injection.
- **3D Scene Core:** Foundational mesh primitives, indexed vertex buffers, PBR materials, and transform hierarchies.
- **Animatable Handles:** Direct numeric and transform controllers (`CameraHandle`, `FloatBufferHandle`) for zero-overhead motion interpolation via `kinetocore`.

---

## Architecture: Core-Facade Separation

```text
[ kinetocore ] (Pure motion/timeline math)
         ▲         ▲
         │         │
[ kinetoxus ]   [ tricore ] (Host-agnostic WGPU render core)
(Dioxus motion)    ▲         ▲
         │       │         │
         ▼       │         │ (Zero Dioxus dependency!)
       [ trioxus ]         └──────► [ SketchUp / CAD Host Plugin ]
  (Dioxus UI Canvas)                (C SDK + Overlay + Readback)
```

---

## Getting Started

Add `tricore` to your `Cargo.toml`:

```toml
[dependencies]
tricore = "0.1"
```

### Headless Offscreen Example

```rust
use tricore::{Camera2D, GpuContext, OffscreenTarget, TargetDimensions};
use wgpu::PowerPreference;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // 1. Initialize standalone headless GPU context
    let gpu = GpuContext::new_headless(PowerPreference::HighPerformance).await?;

    // 2. Create offscreen render target (1920x1080)
    let dimensions = TargetDimensions::new(1920, 1080, 1.0);
    let target = OffscreenTarget::new(
        &gpu.device,
        dimensions,
        wgpu::TextureFormat::Rgba8Unorm,
        Some(wgpu::TextureFormat::Depth32Float),
    );

    // 3. Setup 2D camera
    let camera = Camera2D::new(glam::Vec2::ZERO, 1.0);
    let _view_proj = camera.view_proj_matrix(1920.0, 1080.0);

    println!("Tricore offscreen target ready: {}x{}", target.dimensions.width, target.dimensions.height);
    Ok(())
}
```

---

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.
