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

Concrete implementations: [`hyperball`](https://github.com/arclabs561/hyperball) (Poincare ball, Lorentz hyperboloid). The trait is used by [`flowmatch`](https://github.com/arclabs561/flowmatch) for Riemannian ODE integration.

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

```text
2-simplex [0,1,2]: boundary =
  +1 * [1, 2]
  -1 * [0, 2]
  +1 * [0, 1]

Chain complex check: d(d([0,1,2])) should cancel to zero:
  vertex coefficients: {[2]: 0, [1]: 0, [0]: 0}
  dd = 0? true

3-simplex [0,1,2,3] (tetrahedron): 4 boundary faces
  +1 * [1, 2, 3]
  -1 * [0, 2, 3]
  +1 * [0, 1, 3]
  -1 * [0, 1, 2]
  dd = 0? true
```

## Tests

```bash
cargo test -p skel
```

12 tests covering simplex construction, boundary orientation, chain complex identity (dd = 0), and error handling.

## References

`skel` spans two complementary areas: combinatorial topology (simplices, boundary operators, homology) and differential geometry (the Manifold trait for Riemannian computation).

### Combinatorial topology

- Edelsbrunner & Harer, *Computational Topology: An Introduction* -- standard reference for simplicial homology and persistent homology.
- Hatcher, *Algebraic Topology*, Chapter 2 -- rigorous treatment of simplicial complexes and chain complexes (free online).
- Papillon et al. (2023), "Architectures of Topological Deep Learning" -- taxonomy of neural networks on simplicial complexes; the boundary operator is the core primitive.
- Hajij et al. (2022), "Topological Deep Learning: Going Beyond Graph Data" -- unified framework for learning on higher-order structures.
- Yang & Isufi (2023), "Convolutional Learning on Simplicial Complexes" -- Hodge Laplacian derived from boundary operators.

### Differential geometry / Riemannian computation

- Chen & Lipman (2023), "Riemannian Flow Matching on General Geometries" -- the exp/log/transport interface is exactly the Manifold trait surface.
- de Kruiff et al. (2024), "Pullback Flow Matching on Data Manifolds" -- alternative when closed-form exp/log is unavailable.
- Sherry & Smets (2025), "Flow Matching on Lie Groups" -- suggests a LieGroup subtrait as a future direction.

### Cohomological flows

- Girish et al. (2025), "Persistent Topological Structures and Cohomological Flows" -- theoretical foundation for the CohomologicalFlow trait.
- Maggs et al. (2023), "Simplicial Representation Learning with Neural k-Forms" -- concrete architectures using cochains and coboundary operators.

## License

MIT OR Apache-2.0
