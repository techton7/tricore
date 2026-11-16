<div align="center">
  <img src="assets/icon.svg" alt="tricore logo" width="160" height="160" />
  <h1>tricore</h1>
  <p><strong>Host-agnostic WGPU render core for <code>trioxus</code>, offscreen plugin hosts, and headless Rust pipelines.</strong></p>

  <p>
    <a href="https://crates.io/crates/tricore"><img src="https://img.shields.io/crates/v/tricore.svg" alt="Crates.io" /></a>
    <a href="https://docs.rs/tricore"><img src="https://docs.rs/tricore/badge.svg" alt="Docs.rs" /></a>
    <a href="LICENSE-MIT"><img src="https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg" alt="License" /></a>
  </p>
</div>

---

> **Status:** early-stage crate. The current release establishes the shared crate boundary and foundational types. The renderer surface and API are expected to evolve as `trioxus` and other hosts adopt the core.

## What `tricore` is

`tricore` is the shared graphics core extracted so renderer code does not have to live permanently inside a single host integration crate.

It is intended to support:

- `trioxus`, as the Dioxus-facing canvas and embedding facade
- offscreen/plugin hosts such as SketchUp-style overlay integrations
- headless render, test, and tooling pipelines

`tricore` intentionally does **not** own UI mounting, DOM integration, Blitz widget policy, or host/plugin lifecycle code.

## What is available today

The current crate provides foundational building blocks:

| Module | Current role |
|---|---|
| `renderer` | Shared `GpuContext` for adopted or headless `wgpu::Device` / `wgpu::Queue` ownership |
| `targets` | Render target dimensions plus offscreen color/depth target allocation |
| `readback` | CPU readback staging buffers for color and depth textures |
| `camera` | 2D, 3D, and raw-matrix camera definitions |
| `math` | Host-neutral viewport math such as `viewport_to_world_2d`, `world_to_viewport_2d`, and `viewport_to_ray` |
| `handles` | Renderer-owned camera and numeric buffer handles |
| `batch2d` | Data structures for lines, cubic beziers, grids, and proxy quads |
| `scene3d` | Foundational mesh, material, and transform data structures |

These are **foundations**, not a feature-complete scene engine yet.

## Example

This example uses only APIs that exist in the current crate surface:

```rust
use tricore::{
    glam::vec2,
    viewport_to_world_2d,
    Camera2D,
    Mesh3D,
    ViewportRect,
};

fn main() {
    let camera = Camera2D::default();
    let viewport = ViewportRect::new(0.0, 0.0, 800.0, 600.0);

    let world_pt = viewport_to_world_2d(
        vec2(400.0, 300.0),
        viewport,
        camera.view_proj_matrix(800.0, 600.0).inverse(),
    );

    let cube = Mesh3D::unit_cube();

    assert_eq!(cube.indices.len(), 36);
    println!("Viewport center maps to {world_pt:?}");
}
```

## Design goals

- Keep the core host-agnostic.
- Treat offscreen rendering as a first-class path, not an afterthought.
- Keep viewport-space math in the core and host-specific screen mapping in facade crates.
- Provide reusable building blocks for both 2D batch-heavy and 3D scene workloads.
- Stay compatible with higher-level facades such as `trioxus` without becoming a UI crate itself.

## Not the goal of this crate

At this stage, `tricore` is **not** trying to be:

- a Dioxus UI library
- a SketchUp plugin framework
- a stable 1.0 renderer API
- a Three.js compatibility layer
- a claim of complete rendering feature coverage

## Relationship to sibling crates

- `tricore` owns host-agnostic graphics and math primitives.
- `trioxus` should consume `tricore` to provide Dioxus-facing canvas and embedding APIs.
- Motion ergonomics belong in higher-level motion crates, not in `tricore` itself.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.
