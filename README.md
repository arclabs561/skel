# skel

[![crates.io](https://img.shields.io/crates/v/skel.svg)](https://crates.io/crates/skel)
[![Documentation](https://docs.rs/skel/badge.svg)](https://docs.rs/skel)
[![CI](https://github.com/arclabs561/skel/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/skel/actions/workflows/ci.yml)

Manifold and topology primitives.

## Problem

ODE integrators, gradient descent, and interpolation on curved spaces (spheres, hyperbolic planes, tori) need three operations: step along a geodesic (`exp`), find the direction between two points (`log`), and carry vectors between tangent spaces (`parallel_transport`). These operations vary per geometry, but the code that uses them does not.

`skel` defines the `Manifold` trait so that one ODE integrator or optimizer works on any geometry. It also provides `Simplex` for oriented simplicial complexes with boundary computation.

## Manifold trait

Any Riemannian geometry implements four methods:

```rust
use skel::Manifold;
use ndarray::{Array1, ArrayView1};

struct UnitCircle; // S^1 embedded in R^2

impl Manifold for UnitCircle {
    fn exp_map(&self, x: &ArrayView1<f64>, v: &ArrayView1<f64>) -> Array1<f64> {
        // x is a unit vector on S^1, v is tangent (perpendicular to x).
        // Walk angle = |v| along the circle.
        let angle = v.dot(v).sqrt();
        if angle < 1e-15 { return x.to_owned(); }
        let c = angle.cos();
        let s = angle.sin();
        let dir = v.mapv(|vi| vi / angle);
        &x.mapv(|xi| xi * c) + &dir.mapv(|di| di * s)
    }

    fn log_map(&self, x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> Array1<f64> {
        // Tangent vector at x pointing toward y.
        let dot = x.dot(y).clamp(-1.0, 1.0);
        let angle = dot.acos();
        if angle < 1e-15 { return Array1::zeros(x.len()); }
        let proj = y - &x.mapv(|xi| xi * dot); // component perpendicular to x
        let norm = proj.dot(&proj).sqrt();
        if norm < 1e-15 { return Array1::zeros(x.len()); }
        proj.mapv(|pi| pi * angle / norm)
    }

    fn parallel_transport(
        &self, x: &ArrayView1<f64>, y: &ArrayView1<f64>, v: &ArrayView1<f64>,
    ) -> Array1<f64> {
        // Rotate v from T_x to T_y along the geodesic.
        let log_v = self.log_map(x, y);
        let angle = log_v.dot(&log_v).sqrt();
        if angle < 1e-15 { return v.to_owned(); }
        // For S^1, transport = rotation by the arc angle.
        let c = angle.cos();
        let s = angle.sin();
        let x0 = x[0]; let x1 = x[1];
        let y0 = y[0]; let y1 = y[1];
        // Signed angle from x to y
        let cross = x0 * y1 - x1 * y0;
        let signed = if cross >= 0.0 { angle } else { -angle };
        let cs = signed.cos();
        let sn = signed.sin();
        Array1::from_vec(vec![cs * v[0] - sn * v[1], sn * v[0] + cs * v[1]])
    }

    fn project(&self, x: &ArrayView1<f64>) -> Array1<f64> {
        let norm = x.dot(x).sqrt();
        x.mapv(|xi| xi / norm) // normalize back onto S^1
    }
}
```

Concrete implementations exist for the Poincare ball and Lorentz hyperboloid.

## Simplex

Oriented simplices with boundary computation. The chain complex identity (dd = 0) holds by construction:

```rust
use skel::topology::Simplex;

let tri = Simplex::new_canonical(vec![0, 1, 2]).unwrap();
assert_eq!(tri.dim(), 2);
assert_eq!(tri.boundary().len(), 3); // three oriented edges
```

```bash
cargo run --example simplex_boundary
```

## Tests

```bash
cargo test -p skel
```

12 tests covering simplex construction, boundary orientation, chain complex identity (dd = 0), and error handling.

## References

- Edelsbrunner & Harer, *Computational Topology: An Introduction*
- Hatcher, *Algebraic Topology*, Chapter 2 (simplicial complexes and chain complexes)
- Chen & Lipman (2023), "Riemannian Flow Matching on General Geometries"
- Papillon et al. (2023), "Architectures of Topological Deep Learning"

## License

MIT OR Apache-2.0
