//! Vietoris-Rips filtered complex construction from a distance matrix.
//!
//! The Vietoris-Rips complex \(\mathrm{VR}(X, r)\) contains a simplex
//! \([v_0, \ldots, v_k]\) whenever every pair of its vertices is within
//! distance \(r\):
//!
//! \[
//!   [v_0, \ldots, v_k] \in \mathrm{VR}(X, r)
//!   \iff \max_{i < j} d(v_i, v_j) \leq r
//! \]
//!
//! The filtration value of a simplex is the maximum pairwise distance among
//! its vertices, so the simplex "enters" the complex at exactly the moment
//! all its edges are present.
//!
//! ## Usage
//!
//! ```
//! use skel::vietoris_rips::vietoris_rips;
//!
//! // Three points forming a near-equilateral triangle with side ~1.0.
//! let n = 3;
//! let d = vec![
//!     0.0, 1.0, 1.0,
//!     1.0, 0.0, 1.0,
//!     1.0, 1.0, 0.0,
//! ];
//! let filt = vietoris_rips(&d, n, 2, f64::INFINITY);
//! assert_eq!(filt.complex().len(), 7); // 3 vertices + 3 edges + 1 triangle
//! ```

use crate::filtration::{Filtration, OrdF64};
use crate::topology::Simplex;

/// Build the Vietoris-Rips filtered complex from a distance matrix.
///
/// A `k`-simplex `[v0, ..., vk]` is included when
/// \(\max_{i < j} d(v_i, v_j) \leq \text{max\_radius}\) and \(k \leq\)
/// `max_dim`.  Its filtration value is that maximum pairwise distance.
///
/// # Arguments
///
/// * `distances` - n×n symmetric distance matrix, **row-major** flat layout.
///   `distances[i * n + j]` is the distance from point `i` to point `j`.
///   The diagonal should be 0.  NaN values are not permitted.
/// * `n` - number of points (side length of the distance matrix).
/// * `max_dim` - maximum simplex dimension to include (0 = vertices only,
///   1 = edges, 2 = triangles, ...).
/// * `max_radius` - maximum filtration value; simplices with diameter above
///   this threshold are excluded.
///
/// # Panics
///
/// Panics if `distances.len() != n * n` or if any distance value is NaN.
pub fn vietoris_rips(
    distances: &[f64],
    n: usize,
    max_dim: usize,
    max_radius: f64,
) -> Filtration<OrdF64> {
    assert_eq!(
        distances.len(),
        n * n,
        "distances must be an n×n matrix (n={n}, got len={})",
        distances.len()
    );

    let mut filt: Filtration<OrdF64> = Filtration::new();

    // Always insert the 0-simplices (vertices) at distance 0.
    for v in 0..n {
        filt.insert(Simplex::new_checked(vec![v]).unwrap(), OrdF64::new(0.0));
    }

    if max_dim == 0 {
        return filt;
    }

    // Enumerate candidate simplices dimension by dimension.
    // We build up vertex subsets of size k+1 for k = 1..=max_dim.
    //
    // For efficiency, we iterate over vertex subsets in increasing order.
    // A simplex [v0 < v1 < ... < vk] has diameter = max pairwise distance.
    // We only include it if the diameter <= max_radius.

    // Represent each candidate as a sorted Vec<usize> (= vertices).
    // Build level by level: start from edges (pairs), extend to triangles, etc.

    // Level 1: all edges within max_radius.
    let mut prev_level: Vec<Vec<usize>> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let d = distances[i * n + j];
            assert!(!d.is_nan(), "distance[{i},{j}] is NaN");
            if d <= max_radius {
                let vertices = vec![i, j];
                let simplex = Simplex::new_checked(vertices.clone()).unwrap();
                filt.insert(simplex, OrdF64::new(d));
                prev_level.push(vertices);
            }
        }
    }

    if max_dim == 1 {
        return filt;
    }

    // Levels 2..=max_dim: extend each (k-1)-simplex by appending a vertex
    // larger than all its current vertices, then compute the new diameter.
    for _dim in 2..=max_dim {
        let mut curr_level: Vec<Vec<usize>> = Vec::new();
        for base in &prev_level {
            // The last vertex in the sorted base determines the minimum new vertex.
            let last = *base.last().unwrap();
            for w in (last + 1)..n {
                // Diameter of the new simplex = max(diameter of base, max d(v, w) for v in base).
                let old_diam = diameter(distances, n, base);
                let extension_diam = base
                    .iter()
                    .map(|&v| distances[v * n + w])
                    .fold(0.0f64, f64::max);
                let new_diam = old_diam.max(extension_diam);
                if new_diam <= max_radius {
                    let mut vertices = base.clone();
                    vertices.push(w);
                    let simplex = Simplex::new_checked(vertices.clone()).unwrap();
                    filt.insert(simplex, OrdF64::new(new_diam));
                    curr_level.push(vertices);
                }
            }
        }
        prev_level = curr_level;
        if prev_level.is_empty() {
            break;
        }
    }

    filt
}

/// Diameter (maximum pairwise distance) of a set of vertices.
fn diameter(distances: &[f64], n: usize, vertices: &[usize]) -> f64 {
    let mut max_d = 0.0f64;
    for i in 0..vertices.len() {
        for j in (i + 1)..vertices.len() {
            let d = distances[vertices[i] * n + vertices[j]];
            max_d = max_d.max(d);
        }
    }
    max_d
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::Simplex;
    use proptest::prelude::*;

    fn s(vs: &[usize]) -> Simplex {
        Simplex::new_canonical(vs.to_vec()).unwrap()
    }

    /// Build an equilateral triangle distance matrix.
    fn equilateral(side: f64) -> Vec<f64> {
        vec![0.0, side, side, side, 0.0, side, side, side, 0.0]
    }

    #[test]
    fn three_points_vertices_only() {
        let d = equilateral(1.0);
        let filt = vietoris_rips(&d, 3, 0, f64::INFINITY);
        assert_eq!(filt.complex().len(), 3);
        for v in 0..3 {
            assert!(filt.complex().contains(&s(&[v])));
        }
    }

    #[test]
    fn three_points_equilateral_all_simplices() {
        let d = equilateral(1.0);
        let mut filt = vietoris_rips(&d, 3, 2, f64::INFINITY);
        // 3 vertices + 3 edges + 1 triangle = 7
        assert_eq!(filt.complex().len(), 7);
        assert!(filt.complex().contains(&s(&[0, 1, 2])));
        // Triangle enters at filtration value 1.0
        assert_eq!(filt.value(&s(&[0, 1, 2])).map(|v| v.value()), Some(1.0));
        // Edge filtration values are all 1.0
        assert_eq!(filt.value(&s(&[0, 1])).map(|v| v.value()), Some(1.0));
        // Vertices enter at 0.0
        assert_eq!(filt.value(&s(&[0])).map(|v| v.value()), Some(0.0));
        // Check ordering: vertices before edges before triangle.
        let ord = filt.ordered();
        assert_eq!(ord.len(), 7);
        let tri_pos = ord.iter().position(|s| s.vertices() == [0, 1, 2]).unwrap();
        for edge in &[s(&[0, 1]), s(&[0, 2]), s(&[1, 2])] {
            let edge_pos = ord.iter().position(|s| s == edge).unwrap();
            assert!(edge_pos < tri_pos);
        }
    }

    #[test]
    fn three_points_threshold_excludes_triangle() {
        // Threshold 0.5 < 1.0: no edges or triangles.
        let d = equilateral(1.0);
        let filt = vietoris_rips(&d, 3, 2, 0.5);
        assert_eq!(filt.complex().len(), 3); // vertices only
    }

    #[test]
    fn three_points_threshold_includes_edges_only() {
        // Threshold exactly 1.0: edges present, triangle needs diameter 1.0 so also present.
        let d = equilateral(1.0);
        let filt = vietoris_rips(&d, 3, 1, 1.0);
        assert_eq!(filt.complex().len(), 6); // 3 vertices + 3 edges, no triangle (max_dim=1)
    }

    #[test]
    fn scalene_triangle_filtration_values() {
        // Unequal distances: d(0,1)=1, d(0,2)=2, d(1,2)=3.
        let d = vec![0.0, 1.0, 2.0, 1.0, 0.0, 3.0, 2.0, 3.0, 0.0];
        let mut filt = vietoris_rips(&d, 3, 2, f64::INFINITY);
        // Edge [0,1] enters at 1.0, [0,2] at 2.0, [1,2] at 3.0.
        assert_eq!(filt.value(&s(&[0, 1])).map(|v| v.value()), Some(1.0));
        assert_eq!(filt.value(&s(&[0, 2])).map(|v| v.value()), Some(2.0));
        assert_eq!(filt.value(&s(&[1, 2])).map(|v| v.value()), Some(3.0));
        // Triangle diameter = max(1,2,3) = 3.0.
        assert_eq!(filt.value(&s(&[0, 1, 2])).map(|v| v.value()), Some(3.0));
        // Ordering: [0,1] before [0,2] before [1,2] before triangle.
        let ord = filt.ordered().to_vec();
        let pos = |sv: &[usize]| ord.iter().position(|x| x.vertices() == sv).unwrap();
        assert!(pos(&[0, 1]) < pos(&[0, 2]));
        assert!(pos(&[0, 2]) < pos(&[1, 2]));
        assert!(pos(&[1, 2]) <= pos(&[0, 1, 2]));
    }

    #[test]
    fn four_points_tetrahedron() {
        // 4 equidistant points: complete graph -> tetrahedron filtration.
        let n = 4;
        let mut d = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    d[i * n + j] = 1.0;
                }
            }
        }
        let filt = vietoris_rips(&d, n, 3, f64::INFINITY);
        // 4 vertices + 6 edges + 4 triangles + 1 tet = 15
        assert_eq!(filt.complex().len(), 15);
        assert!(filt.complex().contains(&s(&[0, 1, 2, 3])));
    }

    #[test]
    fn max_dim_respected() {
        let d = equilateral(1.0);
        let filt = vietoris_rips(&d, 3, 1, f64::INFINITY);
        // max_dim=1: no 2-simplices.
        assert_eq!(filt.complex().dimension(), Some(1));
        assert!(!filt.complex().contains(&s(&[0, 1, 2])));
    }

    #[test]
    fn filtration_compatibility_vr() {
        // For every simplex in the VR filtration, all its faces must have
        // filtration value <= the simplex's filtration value.
        let d = vec![
            0.0, 1.0, 2.0, 1.5, 1.0, 0.0, 1.0, 2.5, 2.0, 1.0, 0.0, 1.0, 1.5, 2.5, 1.0, 0.0,
        ];
        let filt = vietoris_rips(&d, 4, 3, f64::INFINITY);
        let all_simplices: Vec<Simplex> = filt
            .complex()
            .f_vector()
            .iter()
            .enumerate()
            .flat_map(|(k, _)| {
                filt.complex()
                    .simplices_of_dim(k)
                    .cloned()
                    .collect::<Vec<_>>()
            })
            .collect();
        for sigma in &all_simplices {
            let val_sigma = *filt.value(sigma).unwrap();
            for (_sign, face) in sigma.boundary() {
                let val_face = *filt.value(&face).unwrap();
                assert!(
                    val_face <= val_sigma,
                    "face {:?} (val {:?}) > coface {:?} (val {:?})",
                    face.vertices(),
                    val_face.value(),
                    sigma.vertices(),
                    val_sigma.value()
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "n×n matrix")]
    fn wrong_distance_matrix_size() {
        vietoris_rips(&[0.0, 1.0, 1.0], 3, 1, f64::INFINITY);
    }

    // --- Property tests ---

    proptest! {
        #[test]
        fn vr_contains_all_vertices(
            side in 0.1f64..5.0,
            n in 2usize..=5,
        ) {
            // Complete equidistant n-point cloud at given side length.
            let mut d = vec![0.0f64; n * n];
            for i in 0..n {
                for j in 0..n {
                    if i != j { d[i * n + j] = side; }
                }
            }
            let filt = vietoris_rips(&d, n, 2, f64::INFINITY);
            for v in 0..n {
                prop_assert!(filt.complex().contains(&s(&[v])));
            }
        }

        #[test]
        fn vr_filtration_compatibility(
            values in proptest::collection::vec(0.1f64..3.0, 9usize..=9),
        ) {
            // Build a 3×3 symmetric matrix from random upper triangle.
            let n = 3;
            let mut d = vec![0.0f64; n * n];
            // Fill in a valid symmetric distance matrix from the first 3 values.
            let pairs = [(0,1),(0,2),(1,2)];
            for (k, &(i, j)) in pairs.iter().enumerate() {
                let v = values[k];
                d[i * n + j] = v;
                d[j * n + i] = v;
            }
            let filt = vietoris_rips(&d, n, 2, f64::INFINITY);
            // Filtration compatibility: faces appear no later than cofaces.
            let all: Vec<Simplex> = (0..3)
                .flat_map(|k| filt.complex().simplices_of_dim(k).cloned().collect::<Vec<_>>())
                .collect();
            for sigma in &all {
                let val_sigma = *filt.value(sigma).unwrap();
                for (_sign, face) in sigma.boundary() {
                    let val_face = *filt.value(&face).unwrap();
                    prop_assert!(val_face <= val_sigma,
                        "face {:?} > coface {:?}", face.vertices(), sigma.vertices());
                }
            }
        }
    }
}
