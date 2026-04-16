//! Simplicial complex: a finite collection of simplices closed under taking faces.
//!
//! ## Construction
//!
//! Use [`SimplicialComplex::insert`] to add simplices with automatic closure
//! (all faces are inserted recursively).  Use [`SimplicialComplex::insert_unchecked`]
//! when building from a source that already guarantees the closure property.
//!
//! ## Topological invariants
//!
//! - [`SimplicialComplex::euler_characteristic`]: \(\chi(K) = \sum_k (-1)^k f_k\)
//! - [`SimplicialComplex::f_vector`]: \((f_0, f_1, \ldots, f_d)\) counts per dimension
//! - [`SimplicialComplex::skeleton`]: the sub-complex \(K^{(k)}\)
//! - [`SimplicialComplex::star`] and [`SimplicialComplex::link`]: local topology around
//!   a face

use crate::topology::Simplex;
use std::collections::HashSet;

/// A simplicial complex: a finite collection of simplices closed under taking faces.
///
/// Simplices are stored stratified by dimension.  `simplices[k]` contains all
/// `k`-simplices.  The closure invariant (every face of a member is also a member)
/// is maintained by [`insert`](SimplicialComplex::insert).
#[derive(Clone, Debug, Default)]
pub struct SimplicialComplex {
    /// `simplices[k]` = all k-simplices in the complex.
    simplices: Vec<HashSet<Simplex>>,
}

impl SimplicialComplex {
    /// Create an empty complex.
    pub fn new() -> Self {
        Self { simplices: vec![] }
    }

    /// Ensure the internal `simplices` vec has at least `dim + 1` layers.
    fn ensure_dim(&mut self, dim: usize) {
        if self.simplices.len() <= dim {
            self.simplices.resize_with(dim + 1, HashSet::new);
        }
    }

    /// Insert a simplex **and all of its faces** (closure).
    ///
    /// This is the safe path: the resulting complex satisfies the closure
    /// property regardless of insertion order.
    pub fn insert(&mut self, simplex: Simplex) {
        let dim = simplex.dim();
        self.ensure_dim(dim);
        // If already present, all faces are already present too.
        if self.simplices[dim].contains(&simplex) {
            return;
        }
        // Recursively insert all boundary faces first.
        for (_sign, face) in simplex.boundary() {
            self.insert(face);
        }
        self.simplices[dim].insert(simplex);
    }

    /// Insert a simplex **without** inserting its faces.
    ///
    /// The caller must guarantee that all faces are already present (or will be
    /// inserted before any operation that relies on the closure property).  Use
    /// this when building from a source that already guarantees validity, to avoid
    /// redundant work.
    pub fn insert_unchecked(&mut self, simplex: Simplex) {
        let dim = simplex.dim();
        self.ensure_dim(dim);
        self.simplices[dim].insert(simplex);
    }

    /// Check if the complex contains a given simplex.
    pub fn contains(&self, simplex: &Simplex) -> bool {
        let dim = simplex.dim();
        self.simplices.get(dim).is_some_and(|s| s.contains(simplex))
    }

    /// All `k`-simplices (unordered).
    pub fn simplices_of_dim(&self, k: usize) -> impl Iterator<Item = &Simplex> {
        self.simplices.get(k).into_iter().flat_map(|s| s.iter())
    }

    /// Maximum dimension of any simplex in the complex, or `None` if empty.
    pub fn dimension(&self) -> Option<usize> {
        self.simplices
            .iter()
            .enumerate()
            .rev()
            .find(|(_, s)| !s.is_empty())
            .map(|(i, _)| i)
    }

    /// Total number of simplices across all dimensions.
    pub fn len(&self) -> usize {
        self.simplices.iter().map(|s| s.len()).sum()
    }

    /// Returns `true` if the complex contains no simplices.
    pub fn is_empty(&self) -> bool {
        self.simplices.iter().all(|s| s.is_empty())
    }

    /// Euler characteristic: \(\chi(K) = \sum_k (-1)^k f_k\).
    ///
    /// For a triangulated surface, \(\chi = V - E + F\).
    pub fn euler_characteristic(&self) -> i64 {
        self.simplices
            .iter()
            .enumerate()
            .map(|(k, s)| {
                let sign: i64 = if k % 2 == 0 { 1 } else { -1 };
                sign * s.len() as i64
            })
            .sum()
    }

    /// f-vector: \((f_0, f_1, \ldots, f_d)\) where \(f_k\) is the number of
    /// `k`-simplices.
    ///
    /// Returns an empty `Vec` for the empty complex.
    pub fn f_vector(&self) -> Vec<usize> {
        // Trim trailing zeros (dimensions with no simplices above the max).
        let top = self.simplices.len();
        let mut fv: Vec<usize> = self.simplices.iter().map(|s| s.len()).collect();
        // Trim from the right: find the last non-zero entry.
        while fv.last() == Some(&0) {
            fv.pop();
        }
        let _ = top; // suppress lint
        fv
    }

    /// The `k`-skeleton: a new complex containing all simplices of dimension ≤ `k`.
    pub fn skeleton(&self, k: usize) -> SimplicialComplex {
        let mut out = SimplicialComplex::new();
        for dim in 0..=k {
            if let Some(layer) = self.simplices.get(dim) {
                for s in layer {
                    out.insert_unchecked(s.clone());
                }
            }
        }
        out
    }

    /// Star of `simplex`: all simplices in the complex that contain `simplex` as a face.
    ///
    /// The star includes `simplex` itself (it is a face of itself in the non-strict sense).
    pub fn star(&self, simplex: &Simplex) -> Vec<&Simplex> {
        let verts = simplex.vertices();
        self.simplices
            .iter()
            .flat_map(|layer| layer.iter())
            .filter(|s| {
                // s contains simplex as a face iff all vertices of simplex appear in s.
                verts.iter().all(|v| s.vertices().contains(v))
            })
            .collect()
    }

    /// Link of `simplex`: faces of simplices in `star(simplex)` that are disjoint
    /// from `simplex`.
    ///
    /// Formally, \(\mathrm{lk}(\sigma) = \{\tau \in K : \tau \cap \sigma = \emptyset,\,
    /// \tau \cup \sigma \in K\}\).
    pub fn link(&self, simplex: &Simplex) -> SimplicialComplex {
        let verts: HashSet<usize> = simplex.vertices().iter().copied().collect();
        let star = self.star(simplex);
        let mut out = SimplicialComplex::new();
        for s in star {
            // Consider all faces (including s itself) that don't intersect simplex.
            self.faces_of(s)
                .into_iter()
                .filter(|f| f.vertices().iter().all(|v| !verts.contains(v)))
                .for_each(|f| out.insert_unchecked(f));
        }
        out
    }

    /// All faces of a simplex that are already in this complex (including itself).
    fn faces_of(&self, simplex: &Simplex) -> Vec<Simplex> {
        let mut result = vec![simplex.clone()];
        let mut stack = vec![simplex.clone()];
        let mut seen: HashSet<Simplex> = HashSet::new();
        seen.insert(simplex.clone());
        while let Some(s) = stack.pop() {
            for (_sign, face) in s.boundary() {
                if self.contains(&face) && seen.insert(face.clone()) {
                    result.push(face.clone());
                    stack.push(face);
                }
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::Simplex;
    use proptest::prelude::*;

    fn simplex(vs: &[usize]) -> Simplex {
        Simplex::new_checked(vs.to_vec()).unwrap()
    }

    #[test]
    fn empty_complex() {
        let k = SimplicialComplex::new();
        assert!(k.is_empty());
        assert_eq!(k.len(), 0);
        assert_eq!(k.dimension(), None);
        assert_eq!(k.euler_characteristic(), 0);
        assert!(k.f_vector().is_empty());
    }

    #[test]
    fn insert_vertex_only() {
        let mut k = SimplicialComplex::new();
        k.insert(simplex(&[0]));
        assert_eq!(k.len(), 1);
        assert_eq!(k.dimension(), Some(0));
        assert!(k.contains(&simplex(&[0])));
        assert_eq!(k.euler_characteristic(), 1);
        assert_eq!(k.f_vector(), vec![1]);
    }

    #[test]
    fn insert_edge_adds_vertices() {
        let mut k = SimplicialComplex::new();
        k.insert(simplex(&[1, 3]));
        // Should contain the edge and both endpoints.
        assert!(k.contains(&simplex(&[1, 3])));
        assert!(k.contains(&simplex(&[1])));
        assert!(k.contains(&simplex(&[3])));
        assert_eq!(k.len(), 3);
        assert_eq!(k.euler_characteristic(), 2 - 1); // 2 vertices - 1 edge = 1
    }

    #[test]
    fn insert_triangle_adds_all_faces() {
        let mut k = SimplicialComplex::new();
        k.insert(simplex(&[0, 1, 2]));
        // 1 triangle + 3 edges + 3 vertices = 7
        assert_eq!(k.len(), 7);
        assert_eq!(k.dimension(), Some(2));
        // Closure: all sub-faces present
        for face in &[
            [0usize, 1, 2].as_ref(),
            &[0, 1],
            &[0, 2],
            &[1, 2],
            &[0],
            &[1],
            &[2],
        ] {
            assert!(k.contains(&Simplex::new_canonical(face.to_vec()).unwrap()));
        }
    }

    #[test]
    fn closure_property_on_double_insert() {
        // Inserting the same simplex twice should be idempotent.
        let mut k = SimplicialComplex::new();
        k.insert(simplex(&[0, 1, 2]));
        k.insert(simplex(&[0, 1, 2]));
        assert_eq!(k.len(), 7);
    }

    #[test]
    fn euler_characteristic_tetrahedron() {
        // A solid tetrahedron: V=4, E=6, F=4, T=1 -> chi = 4-6+4-1 = 1
        let mut k = SimplicialComplex::new();
        k.insert(simplex(&[0, 1, 2, 3]));
        // 4 vertices + 6 edges + 4 triangles + 1 tet = 15
        assert_eq!(k.len(), 15);
        assert_eq!(k.euler_characteristic(), 4 - 6 + 4 - 1);
    }

    #[test]
    fn euler_characteristic_hollow_tetrahedron() {
        // Hollow tetrahedron = just the 4 boundary triangles, no solid interior.
        // V=4, E=6, F=4, T=0 -> chi = 4-6+4 = 2 (sphere)
        let mut k = SimplicialComplex::new();
        for tri in &[[0usize, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]] {
            k.insert(Simplex::new_canonical(tri.to_vec()).unwrap());
        }
        assert_eq!(k.euler_characteristic(), 2);
    }

    #[test]
    fn f_vector_triangle() {
        let mut k = SimplicialComplex::new();
        k.insert(simplex(&[0, 1, 2]));
        assert_eq!(k.f_vector(), vec![3, 3, 1]);
    }

    #[test]
    fn skeleton_extracts_lower_dims() {
        let mut k = SimplicialComplex::new();
        k.insert(simplex(&[0, 1, 2]));
        let sk1 = k.skeleton(1);
        // 1-skeleton: edges and vertices, no triangles.
        assert_eq!(sk1.dimension(), Some(1));
        assert!(!sk1.contains(&simplex(&[0, 1, 2])));
        assert!(sk1.contains(&simplex(&[0, 1])));
        assert!(sk1.contains(&simplex(&[0])));
    }

    #[test]
    fn skeleton_of_full_simplex_at_dim_0() {
        let mut k = SimplicialComplex::new();
        k.insert(simplex(&[0, 1, 2]));
        let sk0 = k.skeleton(0);
        assert_eq!(sk0.dimension(), Some(0));
        assert_eq!(sk0.len(), 3);
    }

    #[test]
    fn star_of_vertex_in_triangle() {
        let mut k = SimplicialComplex::new();
        k.insert(simplex(&[0, 1, 2]));
        let star = k.star(&simplex(&[0]));
        // Star of vertex 0: all simplices containing 0 -> [0],[0,1],[0,2],[0,1,2]
        let star_verts: HashSet<_> = star.iter().map(|s| s.vertices().to_vec()).collect();
        assert!(star_verts.contains(&vec![0usize]));
        assert!(star_verts.contains(&vec![0, 1]));
        assert!(star_verts.contains(&vec![0, 2]));
        assert!(star_verts.contains(&vec![0, 1, 2]));
        assert_eq!(star.len(), 4);
    }

    #[test]
    fn star_of_edge_in_triangle() {
        let mut k = SimplicialComplex::new();
        k.insert(simplex(&[0, 1, 2]));
        let star = k.star(&simplex(&[0, 1]));
        // [0,1] and [0,1,2]
        assert_eq!(star.len(), 2);
    }

    #[test]
    fn link_of_vertex_in_triangle() {
        let mut k = SimplicialComplex::new();
        k.insert(simplex(&[0, 1, 2]));
        let lk = k.link(&simplex(&[0]));
        // Link of 0 in triangle: the edge [1,2] and its faces [1],[2].
        assert!(lk.contains(&simplex(&[1, 2])));
        assert!(lk.contains(&simplex(&[1])));
        assert!(lk.contains(&simplex(&[2])));
        assert!(!lk.contains(&simplex(&[0])));
    }

    #[test]
    fn link_of_edge_in_triangle() {
        let mut k = SimplicialComplex::new();
        k.insert(simplex(&[0, 1, 2]));
        let lk = k.link(&simplex(&[0, 1]));
        // Link of [0,1] in triangle [0,1,2]: just vertex [2].
        assert_eq!(lk.len(), 1);
        assert!(lk.contains(&simplex(&[2])));
    }

    #[test]
    fn simplices_of_dim_iter() {
        let mut k = SimplicialComplex::new();
        k.insert(simplex(&[0, 1, 2]));
        let tris: Vec<_> = k.simplices_of_dim(2).collect();
        assert_eq!(tris.len(), 1);
        assert_eq!(tris[0].vertices(), &[0, 1, 2]);
        let edges: Vec<_> = k.simplices_of_dim(1).collect();
        assert_eq!(edges.len(), 3);
    }

    // --- Property tests ---

    proptest! {
        #[test]
        fn closure_after_insert(mut vs in proptest::collection::vec(0usize..20, 2..5)) {
            vs.sort_unstable();
            vs.dedup();
            prop_assume!(vs.len() >= 2);
            let s = Simplex::new_checked(vs).unwrap();
            let mut k = SimplicialComplex::new();
            k.insert(s.clone());
            // Every face of every simplex in the complex must also be in the complex.
            let all: Vec<Simplex> = k.simplices.iter()
                .flat_map(|layer| layer.iter().cloned())
                .collect();
            for sigma in &all {
                for (_sign, face) in sigma.boundary() {
                    prop_assert!(k.contains(&face),
                        "face {:?} of {:?} not in complex", face.vertices(), sigma.vertices());
                }
            }
        }

        #[test]
        fn euler_characteristic_is_signed_f_vector_sum(mut vs in proptest::collection::vec(0usize..10, 1..5)) {
            vs.sort_unstable();
            vs.dedup();
            prop_assume!(!vs.is_empty());
            let s = Simplex::new_checked(vs).unwrap();
            let mut k = SimplicialComplex::new();
            k.insert(s);
            let chi = k.euler_characteristic();
            let fv = k.f_vector();
            let expected: i64 = fv.iter().enumerate()
                .map(|(i, &f)| if i % 2 == 0 { f as i64 } else { -(f as i64) })
                .sum();
            prop_assert_eq!(chi, expected);
        }

        #[test]
        fn insert_idempotent(mut vs in proptest::collection::vec(0usize..20, 1..5)) {
            vs.sort_unstable();
            vs.dedup();
            prop_assume!(!vs.is_empty());
            let s = Simplex::new_checked(vs).unwrap();
            let mut k = SimplicialComplex::new();
            k.insert(s.clone());
            let len_first = k.len();
            k.insert(s);
            prop_assert_eq!(k.len(), len_first, "insert is not idempotent");
        }
    }
}
