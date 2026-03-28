//! # skel -- Topology and shape primitives
//!
//! This crate provides two families of abstractions:
//!
//! 1. **Combinatorial topology** ([`topology`]) -- simplices, the boundary operator,
//!    and the chain complex identity \(\partial \partial = 0\) that makes homology
//!    well-defined.
//! 2. **Riemannian manifold interface** ([`manifold`] / [`Manifold`]) -- the minimal
//!    trait surface (exponential map, logarithmic map, parallel transport) required
//!    by downstream crates such as `hyp` (hyperbolic geometry).
//!
//! ## Quick start
//!
//! ```
//! use skel::topology::Simplex;
//!
//! // Build a triangle (2-simplex) and inspect its boundary.
//! let tri = Simplex::new_canonical(vec![0, 1, 2]).unwrap();
//! assert_eq!(tri.dim(), 2);
//! assert_eq!(tri.boundary().len(), 3); // three oriented edges
//! ```
//!
//! ## Key mathematical background
//!
//! A **simplicial complex** \(K\) is a finite collection of simplices closed under
//! taking faces.  The **boundary operator** \(\partial_k\) maps each \(k\)-simplex
//! to a signed sum of its \((k{-}1)\)-dimensional faces.  The fundamental identity
//!
//! \[
//!   \partial_{k-1} \circ \partial_k = 0
//! \]
//!
//! guarantees that every boundary is a cycle, which is the algebraic foundation of
//! **homology**: \(H_k = \ker \partial_k \,/\, \operatorname{im} \partial_{k+1}\).
//!
//! ## Modules
//!
//! | Module       | Contents |
//! |--------------|----------|
//! | [`topology`] | [`Simplex`](topology::Simplex), boundary operator, orientation |
//! | [`manifold`] | [`Manifold`] trait (exp/log/transport/project) |
//! | [`lie`]      | SO(3) and SE(3) Lie group exp/log/geodesic interpolation |
//! | [`optim`]    | Riemannian SGD, Adam, geodesic distance |
//! | [`flow`]     | Cohomological flow scaffolding (WIP) |
//! | [`locus`]    | Back-compat shim; prefer `skel::Manifold` |

pub mod flow;
pub mod lie;
pub mod locus; // back-compat shim (prefer `skel::Manifold`)
pub mod manifold;
pub mod optim;
pub mod topology;

// Ergonomic re-exports.
//
// Prefer `skel::Manifold` as the canonical import.
pub use manifold::Manifold;
