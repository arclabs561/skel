# skel

Topology/shape primitives.

`skel` is a small Rust crate with a few topological building blocks (e.g. simplices) and a minimal
`Manifold` trait that other geometry crates can implement.

## Contents

- **`topology`**: `Simplex` and basic combinatorics (e.g. oriented boundaries).
- **`manifold`**: `Manifold` trait (`exp_map`, `log_map`, `parallel_transport`).
- **`flow`**: placeholder scaffolding for flows on complexes.

## Quickstart

```rust
use skel::topology::Simplex;

let s = Simplex::new_canonical(vec![0, 2, 5]).unwrap();
assert_eq!(s.dim(), 2);
assert_eq!(s.boundary().len(), 3);
```

## Status

Experimental: the API is small but not yet treated as stable.

## License

MIT OR Apache-2.0
