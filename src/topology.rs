//! Topological primitives (Simplex, Complex, Homology).
//!
//! # References
//!
//! - Edelsbrunner & Harer, *Computational Topology: An Introduction*.
//! - "Topological simplification of single cell data" (2026 PhD topic).

/// A simplex of dimension k.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Simplex {
    vertices: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SimplexError {
    Empty,
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
    /// Invariant: `vertices` must be sorted and contain no duplicates.
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
    /// This makes canonicalization explicit (we sort), without silently changing semantics (we do
    /// not deduplicate).
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

    pub fn vertices(&self) -> &[usize] {
        &self.vertices
    }

    pub fn dim(&self) -> usize {
        self.vertices.len().saturating_sub(1)
    }

    /// Return the codimension-1 faces with orientation signs \((-1)^i\).
    ///
    /// For a 0-simplex, the boundary is empty.
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
    }
}
