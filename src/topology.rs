//! Combinatorial topology primitives: simplices, boundary operator, orientation.
//!
//! A **\(k\)-simplex** is the convex hull of \(k{+}1\) affinely independent points,
//! but for combinatorial purposes we represent it as an ordered tuple of vertex
//! indices.  The canonical representation keeps vertices in **strictly increasing**
//! order, which fixes the orientation (even permutation of the canonical order).
//!
//! # The boundary operator
//!
//! For a \(k\)-simplex \(\sigma = [v_0, v_1, \ldots, v_k]\), the boundary is:
//!
//! \[
//!   \partial_k \sigma = \sum_{i=0}^{k} (-1)^i \, [v_0, \ldots, \hat{v}_i, \ldots, v_k]
//! \]
//!
//! where \(\hat{v}_i\) means "omit \(v_i\)."  The alternating signs encode
//! **orientation**: each face inherits a consistent orientation from its parent
//! simplex.
//!
//! # The chain complex identity
//!
//! The most important property of the boundary operator is:
//!
//! \[
//!   \partial_{k-1} \circ \partial_k = 0
//! \]
//!
//! This identity (colloquially "\(\partial\partial = 0\)") holds because each
//! codimension-2 face appears in exactly two codimension-1 faces with opposite
//! signs.  It is what makes homology well-defined: every boundary is automatically
//! a cycle, so the quotient \(H_k = \ker \partial_k / \operatorname{im} \partial_{k+1}\)
//! is meaningful.
//!
//! # Why this matters for machine learning
//!
//! The boundary operator is the core algebraic primitive underlying
//! **Topological Deep Learning** (TDL).  Adjacency matrices between
//! \(k\)-simplices are derived from boundary matrices, and the Hodge
//! Laplacian \(L_k = B_k^\top B_k + B_{k+1} B_{k+1}^\top\) (where
//! \(B_k\) is the matrix form of \(\partial_k\)) is the natural
//! generalization of the graph Laplacian to higher-order domains.
//! The \(\partial\partial = 0\) identity ensures that homology is
//! well-defined, which is the algebraic foundation for persistent
//! homology and all downstream topological features used in TDL.
//!
//! # References
//!
//! - Edelsbrunner & Harer, *Computational Topology: An Introduction*.
//! - Hatcher, *Algebraic Topology*, Chapter 2 (free online).
//! - Papillon et al. (2023), "Architectures of Topological Deep Learning:
//!   A Survey of Message-Passing Topological Neural Networks" --
//!   comprehensive taxonomy of neural networks on simplicial/cell/CW
//!   complexes; motivates the boundary operator as the core ML primitive.
//! - Hajij et al. (2022), "Topological Deep Learning: Going Beyond Graph
//!   Data" -- formalizes simplicial complexes as higher-order relational
//!   structures for deep learning, introducing a unified TDL framework.
//! - Yang & Isufi (2023), "Convolutional Learning on Simplicial Complexes"
//!   -- derives adjacency and Hodge Laplacian matrices from boundary
//!   operators; shows that spectral filters on these matrices generalize
//!   GCNs to simplicial domains.

/// An oriented \(k\)-simplex represented by a strictly increasing list of vertex indices.
///
/// A simplex with \(k{+}1\) vertices has **dimension** \(k\):
///
/// | Vertices | Dimension | Geometric name |
/// |----------|-----------|----------------|
/// | 1        | 0         | point          |
/// | 2        | 1         | edge           |
/// | 3        | 2         | triangle       |
/// | 4        | 3         | tetrahedron    |
///
/// # Orientation
///
/// The canonical orientation is the identity permutation of the sorted vertex
/// list.  Even permutations of the vertex order yield the **same** orientation;
/// odd permutations yield the **opposite** orientation.  Since we always store
/// vertices in sorted order, every `Simplex` value carries the canonical
/// (positive) orientation.
///
/// # Invariants
///
/// - `vertices` is non-empty (enforced by constructors).
/// - `vertices` is strictly increasing (sorted, no duplicates).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Simplex {
    vertices: Vec<usize>,
}

/// Error type for simplex construction.
///
/// A valid simplex requires at least one vertex and no duplicate vertices.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SimplexError {
    /// The vertex list was empty.  A simplex must have at least one vertex
    /// (a 0-simplex / point).
    Empty,
    /// The vertex list contained a duplicate.  Each vertex index must appear
    /// at most once, since the vertices of a simplex are by definition distinct.
    DuplicateVertex(usize),
}

impl std::fmt::Display for SimplexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "simplex must have at least one vertex"),
            Self::DuplicateVertex(v) => write!(f, "simplex has duplicate vertex {v}"),
        }
    }
}

impl std::error::Error for SimplexError {}

impl Simplex {
    /// Construct a simplex from a **strictly increasing** vertex list.
    ///
    /// This is the "trust but verify" constructor: it checks that `vertices` is
    /// non-empty and has no consecutive duplicates, and `debug_assert!`s strict
    /// monotonicity.  Use this when you already have sorted input and want to
    /// avoid the cost of sorting.
    ///
    /// # Errors
    ///
    /// - [`SimplexError::Empty`] if `vertices` is empty.
    /// - [`SimplexError::DuplicateVertex`] if any adjacent pair is equal.
    ///
    /// # Panics (debug only)
    ///
    /// Debug-asserts that `vertices` is strictly increasing (not just
    /// duplicate-free).  In release mode this is unchecked, so callers must
    /// guarantee sorted input.
    pub fn new_checked(vertices: Vec<usize>) -> Result<Self, SimplexError> {
        if vertices.is_empty() {
            return Err(SimplexError::Empty);
        }
        for w in vertices.windows(2) {
            if w[0] == w[1] {
                return Err(SimplexError::DuplicateVertex(w[0]));
            }
            debug_assert!(
                w[0] < w[1],
                "new_checked requires strictly increasing vertices"
            );
        }
        Ok(Self { vertices })
    }

    /// Construct a simplex by sorting the vertices and rejecting duplicates.
    ///
    /// This is the "safe" constructor: it sorts the input into canonical
    /// (strictly increasing) order, then checks for duplicates.  It does **not**
    /// silently deduplicate -- duplicate vertices are an error, because they
    /// indicate a malformed simplex specification.
    ///
    /// # Errors
    ///
    /// - [`SimplexError::Empty`] if `vertices` is empty.
    /// - [`SimplexError::DuplicateVertex`] if any vertex appears more than once.
    ///
    /// # Examples
    ///
    /// ```
    /// use skel::topology::Simplex;
    ///
    /// // Unsorted input is fine -- it gets sorted internally.
    /// let s = Simplex::new_canonical(vec![5, 0, 2]).unwrap();
    /// assert_eq!(s.vertices(), &[0, 2, 5]);
    /// ```
    pub fn new_canonical(mut vertices: Vec<usize>) -> Result<Self, SimplexError> {
        if vertices.is_empty() {
            return Err(SimplexError::Empty);
        }
        vertices.sort_unstable();
        for w in vertices.windows(2) {
            if w[0] == w[1] {
                return Err(SimplexError::DuplicateVertex(w[0]));
            }
        }
        Ok(Self { vertices })
    }

    /// Returns the vertex indices in canonical (sorted) order.
    pub fn vertices(&self) -> &[usize] {
        &self.vertices
    }

    /// Returns the combinatorial dimension of the simplex.
    ///
    /// A simplex with \(n\) vertices has dimension \(n - 1\).  A single vertex
    /// (0-simplex) has dimension 0; an edge (1-simplex) has dimension 1, etc.
    ///
    /// Uses `saturating_sub` so the result is always non-negative, though in
    /// practice the vertex list is never empty (enforced by constructors).
    pub fn dim(&self) -> usize {
        self.vertices.len().saturating_sub(1)
    }

    /// Compute the oriented boundary \(\partial \sigma\).
    ///
    /// Returns the codimension-1 faces paired with their orientation signs.
    /// For a \(k\)-simplex \([v_0, \ldots, v_k]\):
    ///
    /// \[
    ///   \partial [v_0, \ldots, v_k] = \sum_{i=0}^{k} (-1)^i \, [v_0, \ldots, \hat{v}_i, \ldots, v_k]
    /// \]
    ///
    /// The sign \((-1)^i\) ensures that applying the boundary operator twice
    /// yields zero (\(\partial\partial = 0\)), because each codimension-2 face
    /// appears exactly twice with opposite signs.
    ///
    /// For a 0-simplex (single vertex), the boundary is empty -- there are no
    /// faces of dimension \(-1\).
    ///
    /// # Returns
    ///
    /// A `Vec` of `(sign, face)` pairs where `sign` is \(+1\) or \(-1\) and
    /// `face` is a `(k{-}1)`-simplex.  The returned faces are in canonical
    /// (sorted vertex) order since removing a vertex from a sorted list
    /// preserves sorted order.
    pub fn boundary(&self) -> Vec<(i32, Simplex)> {
        if self.vertices.len() <= 1 {
            return vec![];
        }
        let mut out = Vec::with_capacity(self.vertices.len());
        for i in 0..self.vertices.len() {
            let mut face = self.vertices.clone();
            face.remove(i);
            // `face` stays sorted and unique because `self.vertices` was.
            let face = Simplex { vertices: face };
            let sign = if i % 2 == 0 { 1 } else { -1 };
            out.push((sign, face));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn simplex_rejects_empty() {
        assert_eq!(
            Simplex::new_checked(vec![]).unwrap_err(),
            SimplexError::Empty
        );
        assert_eq!(
            Simplex::new_canonical(vec![]).unwrap_err(),
            SimplexError::Empty
        );
    }

    #[test]
    fn simplex_rejects_duplicates() {
        assert_eq!(
            Simplex::new_canonical(vec![2, 1, 2]).unwrap_err(),
            SimplexError::DuplicateVertex(2)
        );
    }

    #[test]
    fn boundary_cardinality() {
        let s = Simplex::new_canonical(vec![0, 2, 5]).unwrap();
        let bd = s.boundary();
        assert_eq!(bd.len(), 3);
        for (_sign, face) in bd {
            assert_eq!(face.dim(), 1);
        }
    }

    #[test]
    fn new_checked_accepts_sorted_unique() {
        let s = Simplex::new_checked(vec![0, 2, 5]).unwrap();
        assert_eq!(s.vertices(), &[0, 2, 5]);
        assert_eq!(s.dim(), 2);
    }

    #[test]
    fn new_canonical_sorts_unsorted_input() {
        let s = Simplex::new_canonical(vec![5, 0, 2]).unwrap();
        assert_eq!(s.vertices(), &[0, 2, 5]);
    }

    #[test]
    fn zero_simplex_has_empty_boundary() {
        let s = Simplex::new_checked(vec![7]).unwrap();
        assert_eq!(s.dim(), 0);
        assert!(s.boundary().is_empty());
    }

    #[test]
    fn one_simplex_boundary_has_two_oriented_faces() {
        let s = Simplex::new_checked(vec![1, 3]).unwrap();
        let bd = s.boundary();
        assert_eq!(bd.len(), 2);
        // i=0: remove vertex 1 -> face [3], sign = +1
        assert_eq!(bd[0].0, 1);
        assert_eq!(bd[0].1.vertices(), &[3]);
        // i=1: remove vertex 3 -> face [1], sign = -1
        assert_eq!(bd[1].0, -1);
        assert_eq!(bd[1].1.vertices(), &[1]);
    }

    #[test]
    fn triangle_boundary_orientation_signs() {
        let s = Simplex::new_canonical(vec![0, 1, 2]).unwrap();
        let bd = s.boundary();
        assert_eq!(bd.len(), 3);
        // Signs alternate: +1, -1, +1
        assert_eq!(bd[0].0, 1);
        assert_eq!(bd[0].1.vertices(), &[1, 2]);
        assert_eq!(bd[1].0, -1);
        assert_eq!(bd[1].1.vertices(), &[0, 2]);
        assert_eq!(bd[2].0, 1);
        assert_eq!(bd[2].1.vertices(), &[0, 1]);
    }

    #[test]
    fn simplex_equality_and_hash_consistency() {
        use std::collections::HashSet;
        let s1 = Simplex::new_canonical(vec![5, 0, 2]).unwrap();
        let s2 = Simplex::new_canonical(vec![0, 2, 5]).unwrap();
        let s3 = Simplex::new_canonical(vec![1, 3, 5]).unwrap();
        assert_eq!(s1, s2);
        let mut set = HashSet::new();
        set.insert(s1.clone());
        assert!(set.contains(&s2));
        assert!(!set.contains(&s3));
    }

    #[test]
    fn error_display_format() {
        assert_eq!(
            SimplexError::Empty.to_string(),
            "simplex must have at least one vertex"
        );
        assert_eq!(
            SimplexError::DuplicateVertex(42).to_string(),
            "simplex has duplicate vertex 42"
        );
    }

    // --- Orientation sign tests ---

    #[test]
    fn tetrahedron_boundary_orientation_signs() {
        // 3-simplex [0,1,2,3]: signs should alternate +1, -1, +1, -1
        let s = Simplex::new_checked(vec![0, 1, 2, 3]).unwrap();
        let bd = s.boundary();
        assert_eq!(bd.len(), 4);
        assert_eq!(bd[0].0, 1);   // remove v0 -> [1,2,3]
        assert_eq!(bd[0].1.vertices(), &[1, 2, 3]);
        assert_eq!(bd[1].0, -1);  // remove v1 -> [0,2,3]
        assert_eq!(bd[1].1.vertices(), &[0, 2, 3]);
        assert_eq!(bd[2].0, 1);   // remove v2 -> [0,1,3]
        assert_eq!(bd[2].1.vertices(), &[0, 1, 3]);
        assert_eq!(bd[3].0, -1);  // remove v3 -> [0,1,2]
        assert_eq!(bd[3].1.vertices(), &[0, 1, 2]);
    }

    #[test]
    fn four_simplex_boundary_has_five_faces() {
        // A 4-simplex has 5 boundary faces (3-simplices).
        let s = Simplex::new_checked(vec![0, 1, 2, 3, 4]).unwrap();
        assert_eq!(s.dim(), 4);
        let bd = s.boundary();
        assert_eq!(bd.len(), 5);
        for (sign, face) in &bd {
            assert_eq!(face.dim(), 3);
            assert!(sign.abs() == 1);
        }
        // Signs alternate: +1, -1, +1, -1, +1
        let signs: Vec<i32> = bd.iter().map(|(s, _)| *s).collect();
        assert_eq!(signs, vec![1, -1, 1, -1, 1]);
    }

    // --- Boundary operator: deeper structural properties ---

    #[test]
    fn boundary_faces_are_all_distinct() {
        // All faces in the boundary of a simplex must be distinct.
        let s = Simplex::new_checked(vec![0, 1, 2, 3]).unwrap();
        let bd = s.boundary();
        let face_verts: Vec<&[usize]> = bd.iter().map(|(_, f)| f.vertices()).collect();
        for i in 0..face_verts.len() {
            for j in (i + 1)..face_verts.len() {
                assert_ne!(face_verts[i], face_verts[j], "duplicate faces at {i} and {j}");
            }
        }
    }

    #[test]
    fn boundary_face_vertices_are_subsets_of_parent() {
        // Every face in bd(sigma) should be a subset of sigma's vertices.
        let s = Simplex::new_checked(vec![2, 5, 7, 11]).unwrap();
        let parent_verts = s.vertices();
        for (_, face) in s.boundary() {
            for v in face.vertices() {
                assert!(
                    parent_verts.contains(v),
                    "face vertex {v} not in parent {:?}",
                    parent_verts
                );
            }
        }
    }

    #[test]
    fn dd_zero_triangle_explicit() {
        // Explicitly verify dd = 0 for the standard triangle [0,1,2].
        // bd([0,1,2]) = +[1,2] - [0,2] + [0,1]
        // bd([1,2]) = +[2] - [1]
        // bd([0,2]) = +[2] - [0]
        // bd([0,1]) = +[1] - [0]
        // dd = (+1)(+[2] - [1]) + (-1)(+[2] - [0]) + (+1)(+[1] - [0])
        //    = [2] - [1] - [2] + [0] + [1] - [0]
        //    = 0
        let tri = Simplex::new_checked(vec![0, 1, 2]).unwrap();
        let mut coeffs = std::collections::HashMap::new();
        for (s1, face) in tri.boundary() {
            for (s2, subface) in face.boundary() {
                *coeffs.entry(subface.vertices().to_vec()).or_insert(0i32) += s1 * s2;
            }
        }
        for (verts, coeff) in &coeffs {
            assert_eq!(*coeff, 0, "dd != 0 at vertex {:?}", verts);
        }
    }

    #[test]
    fn dd_zero_tetrahedron_explicit() {
        // Explicitly verify dd = 0 for the standard tetrahedron [0,1,2,3].
        let tet = Simplex::new_checked(vec![0, 1, 2, 3]).unwrap();
        let mut coeffs = std::collections::HashMap::new();
        for (s1, face) in tet.boundary() {
            for (s2, edge) in face.boundary() {
                *coeffs.entry(edge.vertices().to_vec()).or_insert(0i32) += s1 * s2;
            }
        }
        for (verts, coeff) in &coeffs {
            assert_eq!(*coeff, 0, "dd != 0 at edge {:?}", verts);
        }
    }

    #[test]
    fn boundary_signs_are_unit() {
        // All boundary signs must be exactly +1 or -1.
        let s = Simplex::new_checked(vec![0, 1, 2, 3, 4]).unwrap();
        for (sign, _) in s.boundary() {
            assert!(sign == 1 || sign == -1, "non-unit sign: {sign}");
        }
    }

    // --- Edge cases ---

    #[test]
    fn single_vertex_simplex_is_zero_dimensional() {
        let s = Simplex::new_checked(vec![42]).unwrap();
        assert_eq!(s.dim(), 0);
        assert_eq!(s.vertices(), &[42]);
    }

    #[test]
    fn large_vertex_indices() {
        // Simplex with large vertex indices should work fine.
        let s = Simplex::new_checked(vec![1_000_000, 2_000_000, 3_000_000]).unwrap();
        assert_eq!(s.dim(), 2);
        assert_eq!(s.boundary().len(), 3);
    }

    #[test]
    fn new_checked_rejects_duplicates_at_various_positions() {
        // Duplicate at beginning
        assert!(Simplex::new_checked(vec![1, 1, 2]).is_err());
        // Duplicate at end
        assert!(Simplex::new_checked(vec![0, 3, 3]).is_err());
    }

    #[test]
    fn new_canonical_rejects_all_same_vertices() {
        assert_eq!(
            Simplex::new_canonical(vec![5, 5, 5]).unwrap_err(),
            SimplexError::DuplicateVertex(5)
        );
    }

    #[test]
    fn simplex_ord_is_lexicographic() {
        // Ordering should be lexicographic on vertex lists.
        let s1 = Simplex::new_checked(vec![0, 1]).unwrap();
        let s2 = Simplex::new_checked(vec![0, 2]).unwrap();
        let s3 = Simplex::new_checked(vec![1, 2]).unwrap();
        assert!(s1 < s2);
        assert!(s2 < s3);
    }

    #[test]
    fn boundary_of_edge_sums_to_zero_coefficient() {
        // For an edge [a,b], bd = +[b] - [a].  If we assign coefficient 1 to
        // each vertex and sum: (+1) + (-1) = 0.
        let edge = Simplex::new_checked(vec![3, 7]).unwrap();
        let bd = edge.boundary();
        let sum: i32 = bd.iter().map(|(s, _)| *s).sum();
        assert_eq!(sum, 0, "sum of boundary signs should be 0 for a 1-simplex");
    }

    #[test]
    fn boundary_sign_sum_alternates_by_dimension() {
        // For a k-simplex with n = k+1 vertices (k >= 1), the boundary has n
        // faces with signs (-1)^0, (-1)^1, ..., (-1)^{n-1}.
        // Sum = 1 if n is odd, 0 if n is even.
        // (A 0-simplex has empty boundary, so we start from n=2.)
        for n in 2usize..=7 {
            let verts: Vec<usize> = (0..n).collect();
            let s = Simplex::new_checked(verts).unwrap();
            let bd = s.boundary();
            let sign_sum: i32 = bd.iter().map(|(s, _)| *s).sum();
            let expected = if n % 2 == 1 { 1 } else { 0 };
            assert_eq!(sign_sum, expected, "sign sum mismatch for {n}-vertex simplex");
        }
    }

    // --- Property tests ---

    proptest! {
        #[test]
        fn canonical_simplex_has_sorted_unique_vertices(mut vs in proptest::collection::vec(0usize..1000, 1..8)) {
            // Force uniqueness for this property (we're testing postconditions, not error cases).
            vs.sort_unstable();
            vs.dedup();
            prop_assume!(!vs.is_empty());
            let s = Simplex::new_checked(vs.clone()).unwrap();
            prop_assert_eq!(s.vertices(), vs.as_slice());
            prop_assert_eq!(s.dim(), vs.len() - 1);
        }

        #[test]
        fn boundary_boundary_cancels(mut vs in proptest::collection::vec(0usize..100, 2..6)) {
            // The chain complex identity: for each face in bd(sigma), compute its boundary,
            // collect all (sign * subsign, sub-face) pairs, and verify they cancel pairwise.
            vs.sort_unstable();
            vs.dedup();
            prop_assume!(vs.len() >= 2);
            let s = Simplex::new_checked(vs).unwrap();
            let bd1 = s.boundary();

            // Collect all signed sub-faces from bd(bd(s))
            let mut signed_subfaces: Vec<(i32, Vec<usize>)> = Vec::new();
            for (sign1, face) in &bd1 {
                for (sign2, subface) in face.boundary() {
                    signed_subfaces.push((sign1 * sign2, subface.vertices().to_vec()));
                }
            }

            // Group by vertex set and verify cancellation (sum of signs = 0 for each).
            signed_subfaces.sort_by(|a, b| a.1.cmp(&b.1));
            let mut i = 0;
            while i < signed_subfaces.len() {
                let key = &signed_subfaces[i].1;
                let mut sum = 0i32;
                let mut j = i;
                while j < signed_subfaces.len() && signed_subfaces[j].1 == *key {
                    sum += signed_subfaces[j].0;
                    j += 1;
                }
                prop_assert_eq!(sum, 0, "dd != 0 for subface {:?}", key);
                i = j;
            }
        }

        #[test]
        fn boundary_cardinality_equals_vertex_count(mut vs in proptest::collection::vec(0usize..200, 1..8)) {
            // A k-simplex (k+1 vertices) has exactly k+1 boundary faces (or 0 if k=0).
            vs.sort_unstable();
            vs.dedup();
            prop_assume!(!vs.is_empty());
            let n = vs.len();
            let s = Simplex::new_checked(vs).unwrap();
            let bd = s.boundary();
            if n == 1 {
                prop_assert!(bd.is_empty());
            } else {
                prop_assert_eq!(bd.len(), n);
            }
        }

        #[test]
        fn boundary_faces_have_correct_dimension(mut vs in proptest::collection::vec(0usize..200, 2..8)) {
            // Each face in bd(sigma) has dimension exactly dim(sigma) - 1.
            vs.sort_unstable();
            vs.dedup();
            prop_assume!(vs.len() >= 2);
            let s = Simplex::new_checked(vs).unwrap();
            let d = s.dim();
            for (_, face) in s.boundary() {
                prop_assert_eq!(face.dim(), d - 1);
            }
        }

        #[test]
        fn boundary_faces_are_sorted(mut vs in proptest::collection::vec(0usize..200, 2..8)) {
            // Each boundary face should have sorted vertices (inherited from parent).
            vs.sort_unstable();
            vs.dedup();
            prop_assume!(vs.len() >= 2);
            let s = Simplex::new_checked(vs).unwrap();
            for (_, face) in s.boundary() {
                let fv = face.vertices();
                for w in fv.windows(2) {
                    prop_assert!(w[0] < w[1], "face vertices not sorted: {:?}", fv);
                }
            }
        }

        #[test]
        fn new_canonical_and_new_checked_agree(mut vs in proptest::collection::vec(0usize..500, 1..8)) {
            // After dedup+sort, both constructors should produce the same simplex.
            vs.sort_unstable();
            vs.dedup();
            prop_assume!(!vs.is_empty());
            let checked = Simplex::new_checked(vs.clone()).unwrap();
            let canonical = Simplex::new_canonical(vs).unwrap();
            prop_assert_eq!(checked, canonical);
        }
    }
}

/// Tests for the [`Manifold`](crate::Manifold) trait default implementation.
#[cfg(test)]
mod manifold_tests {
    use crate::Manifold;
    use ndarray::array;

    /// A trivial Euclidean manifold for testing default impls and the trait surface.
    struct EuclideanManifold;

    impl Manifold for EuclideanManifold {
        fn exp_map(
            &self,
            x: &ndarray::ArrayView1<f64>,
            v: &ndarray::ArrayView1<f64>,
        ) -> ndarray::Array1<f64> {
            x + v
        }

        fn log_map(
            &self,
            x: &ndarray::ArrayView1<f64>,
            y: &ndarray::ArrayView1<f64>,
        ) -> ndarray::Array1<f64> {
            y - x
        }

        fn parallel_transport(
            &self,
            _x: &ndarray::ArrayView1<f64>,
            _y: &ndarray::ArrayView1<f64>,
            v: &ndarray::ArrayView1<f64>,
        ) -> ndarray::Array1<f64> {
            // In Euclidean space, parallel transport is the identity.
            v.to_owned()
        }
    }

    #[test]
    fn euclidean_exp_log_roundtrip() {
        let m = EuclideanManifold;
        let x = array![1.0, 2.0, 3.0];
        let v = array![0.5, -1.0, 0.0];
        let y = m.exp_map(&x.view(), &v.view());
        let v_back = m.log_map(&x.view(), &y.view());
        for (a, b) in v.iter().zip(v_back.iter()) {
            assert!((a - b).abs() < 1e-12, "exp/log roundtrip failed");
        }
    }

    #[test]
    fn euclidean_log_exp_roundtrip() {
        let m = EuclideanManifold;
        let x = array![0.0, 0.0];
        let y = array![3.0, 4.0];
        let v = m.log_map(&x.view(), &y.view());
        let y_back = m.exp_map(&x.view(), &v.view());
        for (a, b) in y.iter().zip(y_back.iter()) {
            assert!((a - b).abs() < 1e-12, "log/exp roundtrip failed");
        }
    }

    #[test]
    fn euclidean_parallel_transport_preserves_norm() {
        let m = EuclideanManifold;
        let x = array![1.0, 0.0];
        let y = array![0.0, 1.0];
        let v = array![3.0, 4.0];
        let w = m.parallel_transport(&x.view(), &y.view(), &v.view());
        let norm_v: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_w: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            (norm_v - norm_w).abs() < 1e-12,
            "parallel transport changed norm: {norm_v} vs {norm_w}"
        );
    }

    #[test]
    fn default_project_is_identity() {
        let m = EuclideanManifold;
        let x = array![1.0, 2.0, 3.0];
        let projected = m.project(&x.view());
        assert_eq!(x, projected);
    }

    #[test]
    fn euclidean_log_map_gives_geodesic_distance() {
        let m = EuclideanManifold;
        let x = array![0.0, 0.0];
        let y = array![3.0, 4.0];
        let v = m.log_map(&x.view(), &y.view());
        let dist: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((dist - 5.0).abs() < 1e-12, "distance should be 5.0, got {dist}");
    }
}
