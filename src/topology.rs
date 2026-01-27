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
    }
}
