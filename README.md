# skel

Topology and manifold primitives for computational geometry.

## What it provides

- **`Simplex`**: Oriented simplices with boundary computation (chain complex `dd = 0`).
- **`Manifold` trait**: `exp_map`, `log_map`, `parallel_transport`, `project`. Implemented by [`hyperball`](https://github.com/arclabs561/hyp) (Poincare ball) and usable by any Riemannian geometry crate.

## Usage

```toml
[dependencies]
skel = "0.1.0"
```

```rust
use skel::topology::Simplex;

// A triangle [0, 2, 5] has 3 oriented edges as boundary.
let s = Simplex::new_canonical(vec![0, 2, 5]).unwrap();
assert_eq!(s.dim(), 2);
assert_eq!(s.boundary().len(), 3);
```

Implementing a manifold:

```rust
use skel::Manifold;
use ndarray::{Array1, ArrayView1};

struct MySphere;

impl Manifold for MySphere {
    fn exp_map(&self, x: &ArrayView1<f64>, v: &ArrayView1<f64>) -> Array1<f64> {
        // geodesic step from x in direction v
        todo!()
    }
    fn log_map(&self, x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> Array1<f64> {
        // inverse: tangent vector from x toward y
        todo!()
    }
    fn parallel_transport(&self, x: &ArrayView1<f64>, y: &ArrayView1<f64>, v: &ArrayView1<f64>) -> Array1<f64> {
        // move tangent vector v from T_x to T_y along geodesic
        todo!()
    }
}
```

## Examples

```bash
cargo run --example simplex_boundary  # boundary computation, dd=0 chain complex identity
```

## Tests

```bash
cargo test -p skel
```

12 tests covering simplex construction, boundary orientation, chain complex identity (`dd = 0`), and error handling.

## License

MIT OR Apache-2.0
