# Examples

Topology and manifold primitives.

## Combinatorial Topology

| Example | What It Shows |
|---------|---------------|
| `simplex_boundary` | Oriented boundary operator, dd=0 identity for triangle and tetrahedron |
| `simplicial_complex` | Boundary matrices, dd=0 verification, Betti numbers via rank-nullity |
| `vietoris_rips` | Vietoris-Rips complex from a 2D point cloud, Betti numbers across epsilon sweep |

```sh
cargo run --example simplex_boundary
cargo run --example simplicial_complex
cargo run --example vietoris_rips
```

The `simplicial_complex` example builds the surface of a tetrahedron (homeomorphic to S^2), constructs the full chain complex, and computes Betti numbers b_0=1, b_1=0, b_2=1.

The `vietoris_rips` example generates 32 points from two clusters, builds the Vietoris-Rips complex (edges for pairwise distance <= epsilon, triangles for cliques), and prints a table of edge/triangle counts and Betti numbers b_0, b_1 as epsilon increases from 0.5 to 5.0.

## Manifold Trait

| Example | What It Shows |
|---------|---------------|
| `manifold_circle` | `Manifold` impl for S^1: exp/log round-trip, geodesic interpolation, parallel transport |

```sh
cargo run --example manifold_circle
```

The `Manifold` trait (exp_map, log_map, parallel_transport) is the interface consumed by `hyperball` (hyperbolic geometry) and `flowmatch` (Riemannian ODE integration). The circle example is a minimal implementation that verifies the key identities.
