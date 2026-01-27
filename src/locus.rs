//! Back-compat shim for older import paths.
//!
//! Historically, `Manifold` lived at `skel::locus::manifold::Manifold`.
//! The canonical location is now `skel::Manifold` / `skel::manifold::Manifold`.

pub mod manifold {
    pub use crate::manifold::Manifold;
}
