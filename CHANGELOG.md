# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Source history (unversioned)

These source changes are listed without assigning them to a published release.

### Fixed

- Validate every Vietoris–Rips distance-matrix entry for NaN before construction, including diagonal/lower-triangle entries and vertices-only calls. Infinite distances remain supported.
  ([source change](https://github.com/arclabs561/skel/commit/c63eed8c61d32441d596389320d465b85d8057a2))

## [0.1.7] - 2026-04-20

### Added
- Topological data analysis: `SimplicialComplex`, `Filtration`, and Vietoris-Rips construction.
- `OrdF64` ordered-float helper.

### Deprecated
- `optim` module (moved to `descend::riemannian`).

### Fixed
- Clippy lints in examples.

## [0.1.4] - 2026-04-06

### Added
- SO(3) and SE(3) Lie group primitives.
- Riemannian SGD and Adam optimizers.
- `[workspace]` table for standalone builds.

### Changed
- Updated the Lie algebra module.
- Condensed the README to prose and trimmed feature lists.

## [0.1.0] - 2026-02-09

### Added
- Initial release: manifold primitives and re-exports.
- `Manifold` trait with a default `project()` method.
- Simplicial complex and Vietoris-Rips examples.
- Math formulas in the rustdocs.
