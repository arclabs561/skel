//! The [`Manifold`] trait: a minimal Riemannian manifold interface.
//!
//! A **Riemannian manifold** \((M, g)\) is a smooth manifold equipped with a
//! metric tensor that defines lengths, angles, and geodesics.  This trait
//! captures the three operations that downstream code needs for
//! manifold-valued computation:
//!
//! | Operation            | Signature                     | Intuition                                        |
//! |----------------------|-------------------------------|--------------------------------------------------|
//! | Exponential map      | \(\exp_x(v) \to y\)          | "Walk from \(x\) in direction \(v\)"             |
//! | Logarithmic map      | \(\log_x(y) \to v\)          | "Which direction at \(x\) leads to \(y\)?"       |
//! | Parallel transport   | \(\Gamma_{x \to y}(v) \to w\)| "Carry \(v\) from \(T_xM\) to \(T_yM\)"         |
//!
//! Together, \(\exp\) and \(\log\) form an approximate local inverse pair:
//!
//! \[
//!   \exp_x(\log_x(y)) = y, \qquad \log_x(\exp_x(v)) = v
//! \]
//!
//! when \(y\) is within the injectivity radius of \(x\).
//!
//! # Ownership
//!
//! `skel` owns this trait definition.  Concrete implementations (hyperbolic,
//! spherical, Euclidean, etc.) live in downstream crates (e.g. `hyperball`).
//!
//! # Motivation
//!
//! The `exp`/`log`/`transport` surface is exactly the interface required by
//! **Riemannian Flow Matching** (RFM): the ODE integrator steps along
//! geodesics via `exp`, computes velocity fields via `log`, and moves
//! tangent vectors between steps via `parallel_transport`.  Any geometry
//! that implements this trait can be used as the base manifold for RFM
//! without changes to the flow-matching code.
//!
//! # References
//!
//! - Chen & Lipman (2023), "Riemannian Flow Matching on General
//!   Geometries" -- the exp/log/transport surface defined here is
//!   exactly the manifold interface RFM requires.
//! - de Kruiff et al. (2024), "Pullback Flow Matching on Data Manifolds"
//!   -- an alternative when closed-form exp/log is unavailable; works
//!   with learned pullback maps instead.
//! - Sherry & Smets (2025), "Flow Matching on Lie Groups" -- suggests
//!   a `LieGroup` subtrait as a future direction, exploiting group
//!   structure for more efficient exp/log and transport.

use ndarray::{Array1, ArrayView1};

/// Minimal Riemannian manifold interface for manifold-valued computation.
///
/// Implementors provide the exponential map, logarithmic map, and parallel
/// transport -- the three primitives sufficient for gradient descent on
/// manifolds, geodesic interpolation, and ODE integration on curved spaces.
///
/// # Key identities
///
/// For points within the **injectivity radius**:
///
/// - **Round-trip**: \(\exp_x(\log_x(y)) = y\)
/// - **Inverse**: \(\log_x(\exp_x(v)) = v\)
/// - **Isometry**: parallel transport preserves inner products:
///   \(\langle \Gamma_{x \to y} u, \Gamma_{x \to y} v \rangle_y = \langle u, v \rangle_x\)
///
/// # Default implementations
///
/// - [`project`](Manifold::project) defaults to the identity, which is correct
///   for unconstrained Euclidean space.  Override for constrained manifolds.
pub trait Manifold {
    /// Map a tangent vector to a manifold point: \(\exp_x(v) \in M\).
    ///
    /// Given a base point \(x \in M\) and a tangent vector \(v \in T_x M\),
    /// returns the point reached by following the geodesic from \(x\) in
    /// direction \(v\) for unit time.
    ///
    /// The length of the geodesic segment equals \(\|v\|_x\) (the norm
    /// induced by the Riemannian metric at \(x\)).
    fn exp_map(&self, x: &ArrayView1<f64>, v: &ArrayView1<f64>) -> Array1<f64>;

    /// Map a manifold point to a tangent vector: \(\log_x(y) \in T_x M\).
    ///
    /// This is the (local) inverse of [`exp_map`](Manifold::exp_map).  Given
    /// two manifold points \(x, y \in M\), returns the tangent vector at \(x\)
    /// whose exponential map reaches \(y\).
    ///
    /// The norm \(\|\log_x(y)\|_x\) equals the geodesic distance \(d(x, y)\).
    ///
    /// # Preconditions
    ///
    /// \(y\) must be within the injectivity radius of \(x\) for the result
    /// to be unique.
    fn log_map(&self, x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> Array1<f64>;

    /// Parallel transport a tangent vector along a geodesic:
    /// \(\Gamma_{x \to y}(v) \in T_y M\).
    ///
    /// Moves the tangent vector \(v \in T_x M\) to the tangent space at \(y\)
    /// along the unique geodesic connecting \(x\) and \(y\), preserving the
    /// Riemannian inner product:
    ///
    /// \[
    ///   \langle \Gamma_{x \to y} u,\; \Gamma_{x \to y} v \rangle_y
    ///   = \langle u, v \rangle_x
    /// \]
    ///
    /// This is essential for comparing or accumulating tangent vectors that live
    /// in different tangent spaces (e.g., Riemannian SGD, vector field
    /// integration).
    fn parallel_transport(
        &self,
        x: &ArrayView1<f64>,
        y: &ArrayView1<f64>,
        v: &ArrayView1<f64>,
    ) -> Array1<f64>;

    /// Project a point in ambient space back onto the manifold.
    ///
    /// Numerical drift during ODE integration (repeated [`exp_map`](Manifold::exp_map)
    /// steps) can push points slightly off the manifold surface.  This method
    /// corrects that drift by finding the nearest point on \(M\).
    ///
    /// # Default implementation
    ///
    /// Returns `x` unchanged (the identity projection), which is correct for
    /// unconstrained Euclidean spaces where every point in the ambient space is
    /// already on the manifold.
    ///
    /// Manifolds embedded in a higher-dimensional space with constraints
    /// (spheres: \(\|x\| = 1\); Poincare balls: \(\|x\| < 1\); etc.) **must**
    /// override this method.
    fn project(&self, x: &ArrayView1<f64>) -> Array1<f64> {
        x.to_owned()
    }
}
