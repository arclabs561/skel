use ndarray::{Array1, ArrayView1};

/// Minimal manifold interface used across the workspace.
///
/// `skel` owns this trait; specific manifold implementations live in other crates
/// (e.g. `hyp` for hyperbolic geometry).
pub trait Manifold {
    /// Map a tangent vector `v ∈ T_x M` to the manifold point `exp_x(v)`.
    fn exp_map(&self, x: &ArrayView1<f64>, v: &ArrayView1<f64>) -> Array1<f64>;

    /// Map a manifold point `y ∈ M` to a tangent vector `log_x(y) ∈ T_x M`.
    fn log_map(&self, x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> Array1<f64>;

    /// Parallel transport a tangent vector `v ∈ T_x M` to `T_y M` along the geodesic from `x` to `y`.
    fn parallel_transport(
        &self,
        x: &ArrayView1<f64>,
        y: &ArrayView1<f64>,
        v: &ArrayView1<f64>,
    ) -> Array1<f64>;

    /// Project a point in ambient space back onto the manifold.
    ///
    /// Numerical drift during ODE integration (repeated exp_map steps) can push
    /// points slightly off the manifold.  This method corrects that drift.
    ///
    /// The default implementation is the identity (assumes the point is already on
    /// the manifold), which is correct for unconstrained Euclidean spaces.
    /// Manifolds with constraints (spheres, Poincare balls, etc.) should override.
    fn project(&self, x: &ArrayView1<f64>) -> Array1<f64> {
        x.to_owned()
    }
}
