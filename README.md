# skel

[![crates.io](https://img.shields.io/crates/v/skel.svg)](https://crates.io/crates/skel)
[![Documentation](https://docs.rs/skel/badge.svg)](https://docs.rs/skel)

Topology and manifold primitives for topological data analysis (TDA) and
geometric machine learning.

## Modules

| Module | What it does |
|--------|-------------|
| `topology` | Oriented simplices, boundary operator, chain complex identity |
| `complex` | Simplicial complex container with star, link, skeleton, Euler characteristic |
| `filtration` | Filtered simplicial complexes, boundary matrix output for persistence algorithms |
| `vietoris_rips` | Vietoris-Rips complex construction from distance matrices |
| `Manifold` trait | Riemannian geometry: exp, log, parallel transport, projection |
| `lie` | SO(3), SE(3) Lie groups |

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

let filt = vietoris_rips(&distances, 3, 2, 2.0);

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

Any Riemannian geometry implements exp, log, parallel transport, and projection:

```rust
use skel::Manifold;
```

Concrete implementations exist for the Poincare ball, Lorentz hyperboloid,
SO(3), and SE(3).

## Boundary matrix format

`Filtration::boundary_matrix()` returns `Vec<Vec<(i32, usize)>>` -- sparse
columns of (coefficient, row_index) pairs. This is compatible with
[lophat](https://crates.io/crates/lophat) and
[phlite](https://crates.io/crates/phlite) for computing persistent homology.

## License

MIT OR Apache-2.0
