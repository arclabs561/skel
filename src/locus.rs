//! Back-compatibility shim for older import paths.
//!
//! Historically, the [`Manifold`](crate::Manifold) trait lived at
//! `skel::locus::manifold::Manifold`.  The canonical location is now
//! [`skel::Manifold`](crate::Manifold) (re-exported from [`crate::manifold`]).
//!
//! This module exists solely to avoid breaking downstream code that still uses
//! the old path.  New code should import from `skel::Manifold` directly.

/// Re-export of [`crate::manifold`] for path compatibility.
pub mod manifold {
    pub use crate::manifold::Manifold;
}
