//! skel: Topology and shape primitives
//!
//! This crate provides topological primitives (simplices, complexes) and
//! the minimal manifold trait used across the workspace.

pub mod flow;
pub mod locus; // back-compat shim (prefer `skel::Manifold`)
pub mod manifold;
pub mod topology;

// Ergonomic re-exports.
//
// Prefer `skel::Manifold` as the canonical import.
pub use manifold::Manifold;
