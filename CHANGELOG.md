# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1](https://github.com/techton7/tricore/compare/v0.1.0...v0.1.1) - 2026-11-25

### Added

- *(preview3d)* add Pass A0 opaque depth pre-pass, smooth Porter-Duff OIT, and ULP self-peel guard
- *(preview3d)* add depth-stamp mode and K=4 stochastic depth peeling with blue-noise resolve
- *(tricore)* add CompositeFinalizePipeline, Gizmo depth stamp, ChannelSwizzle, and DepthReadbackSurface
- *(tricore)* upgrade to wgpu 30 and add offscreen, readback, Camera2D, and 3D/Gizmo/blit passes

### Other

- *(tricore)* align README with current crate surface
