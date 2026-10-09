# skel

[![crates.io](https://img.shields.io/crates/v/skel.svg)](https://crates.io/crates/skel)
[![Documentation](https://docs.rs/skel/badge.svg)](https://docs.rs/skel)

Topology and manifold primitives.

## Modules

| Module | What it does |
|--------|-------------|
| `topology` | Oriented simplices, boundary operator, chain complex identity (`dd=0`) |
| `complex` | Simplicial complex with star, link, skeleton, f-vector, Euler characteristic |
| `filtration` | Filtered simplicial complexes, boundary matrix output for persistence |
| `vietoris_rips` | Vietoris-Rips complex from distance matrices |
| `Manifold` trait | Riemannian geometry: exp, log, parallel transport, projection |
| `lie` | SO(3), SE(3) exp/log and geodesic interpolation as free functions |
| `optim` | Riemannian SGD and Adam on manifolds (deprecated copy; moved to `descend::riemannian`) |

## TDA quickstart

```rust
use skel::vietoris_rips::vietoris_rips;

// Three points forming an equilateral triangle
let d = 1.0;
let distances = vec![
    0.0, d,   d,
    d,   0.0, d,
    d,   d,   0.0,
];

let mut filt = vietoris_rips(&distances, 3, 2, 2.0);

// 3 vertices + 3 edges + 1 triangle = 7 simplices
assert_eq!(filt.complex().len(), 7);
assert_eq!(filt.complex().euler_characteristic(), 1); // disk

// Boundary matrix ready for persistent homology (lophat, phlite)
let bm = filt.boundary_matrix();
assert_eq!(bm.len(), 7);
```

## Simplicial complex

```rust
use skel::topology::Simplex;
use skel::complex::SimplicialComplex;

let mut k = SimplicialComplex::new();
let tri = Simplex::new_canonical(vec![0, 1, 2]).unwrap();
k.insert(tri); // automatically inserts all faces

assert_eq!(k.f_vector(), vec![3, 3, 1]); // 3 vertices, 3 edges, 1 triangle
assert_eq!(k.euler_characteristic(), 1);
```

## Manifold trait

A Riemannian geometry implements exp, log, parallel transport, and projection.
`skel` defines the trait; it ships no geometry implementations outside tests and
examples. The Poincare ball implementation lives in
[`hyperball`](https://crates.io/crates/hyperball) (`PoincareBall<f64>`). The
`lie` module's SO(3) and SE(3) maps are free functions on fixed-size arrays,
not `Manifold` implementations. Riemannian optimizers (SGD, Adam) that work
with any `Manifold` implementation live in the `descend` crate
(`descend::riemannian`); the in-crate `optim` module is a deprecated copy kept
for compatibility.

## Boundary matrix format

`Filtration::boundary_matrix()` returns `Vec<Vec<(i32, usize)>>` -- sparse
columns of (coefficient, row_index) pairs. This is compatible with
[lophat](https://crates.io/crates/lophat) and
[phlite](https://crates.io/crates/phlite) for computing persistent homology.

## Examples

All examples in `examples/`. Run with `cargo run --example <name>`.

| Example | What it shows |
|---------|---------------|
| `simplex_boundary` | Oriented boundary operator, `dd=0` identity for triangle and tetrahedron |
| `simplicial_complex` | Boundary matrices, Betti numbers of a tetrahedron surface (S^2) |
| `vietoris_rips` | Build VR complexes from a 2D point cloud, sweep epsilon to see topology change |
| `manifold_circle` | `Manifold` impl for S^1: exp/log round-trip, geodesic interpolation |
| `riemannian_optimization` | The deprecated in-crate Riemannian SGD and Adam |
| `persistence_reduction` | Persistent homology by reducing a filtration boundary matrix |
| `rotor_rotation` | Geometric-algebra rotor rotation in 2D |

## License

MIT OR Apache-2.0
