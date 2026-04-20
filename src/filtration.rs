//! Filtered simplicial complex for persistent homology.
//!
//! A **filtration** assigns a value \(f(\sigma) \in V\) to each simplex
//! \(\sigma\), where \(V\) is a totally ordered type, such that faces always
//! enter no later than the simplices they bound:
//!
//! \[
//!   \tau \text{ is a face of } \sigma \implies f(\tau) \leq f(\sigma)
//! \]
//!
//! The [`Filtration::ordered`] method returns simplices in non-decreasing
//! filtration order, breaking ties by (dimension, lexicographic vertex order)
//! to give a stable total order compatible with the boundary operator.
//!
//! ## Boundary matrix
//!
//! [`Filtration::boundary_matrix`] returns the boundary matrix in column form:
//! each column `j` contains the (coefficient, row_index) pairs for
//! \(\partial \sigma_j\) with rows indexed by the same filtration order.
//! This is the input format expected by persistence algorithms such as
//! `lophat` and `phlite`.

use crate::complex::SimplicialComplex;
use crate::topology::Simplex;
use std::collections::HashMap;

/// A totally-ordered wrapper around `f64` using total bit-pattern ordering.
///
/// `f64` does not implement `Eq` or `Ord` because of NaN, so it cannot be
/// used directly as a filtration value.  `OrdF64` wraps an `f64` and derives
/// total order via `f64::to_bits`, which gives NaN a deterministic (if
/// mathematically arbitrary) position.  In practice, filtration values should
/// never be NaN; callers should validate inputs before inserting.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OrdF64(u64);

impl OrdF64 {
    /// Wrap a finite (non-NaN) `f64` filtration value.
    ///
    /// # Panics
    ///
    /// Panics if `v` is NaN.
    pub fn new(v: f64) -> Self {
        assert!(!v.is_nan(), "filtration value must not be NaN");
        // IEEE 754 total order: negate if negative so the bit pattern sorts correctly.
        // For non-negative values, to_bits preserves order.
        // For negative values, the sign bit is set, so we need to flip all bits.
        let bits = v.to_bits();
        let ordered = if bits >> 63 == 0 {
            bits | (1u64 << 63) // positive: set sign bit so positives > negatives
        } else {
            !bits // negative: flip all bits so more-negative < less-negative
        };
        Self(ordered)
    }

    /// Recover the underlying `f64`.
    pub fn value(self) -> f64 {
        // Reverse the transformation from `new`.
        let bits = if self.0 >> 63 == 1 && (self.0 & ((1u64 << 63) - 1)) != 0 {
            // Was positive: clear the sign bit we added.
            self.0 & !(1u64 << 63)
        } else if self.0 >> 63 == 0 {
            // Was negative: flip all bits back.
            !self.0
        } else {
            // Was exactly 0.0 (positive zero: bits = 0x0000..., stored as 0x8000...)
            self.0 & !(1u64 << 63)
        };
        f64::from_bits(bits)
    }
}

impl PartialOrd for OrdF64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

impl From<f64> for OrdF64 {
    fn from(v: f64) -> Self {
        Self::new(v)
    }
}

impl From<OrdF64> for f64 {
    fn from(v: OrdF64) -> Self {
        v.value()
    }
}

/// A filtered simplicial complex: each simplex has a filtration value, and
/// simplices are ordered non-decreasingly by that value.
///
/// The type parameter `V` is the filtration value type (must implement `Ord + Clone`).
/// Use [`OrdF64`] for real-valued filtrations (e.g. Vietoris-Rips).
pub struct Filtration<V: Ord + Clone> {
    complex: SimplicialComplex,
    values: HashMap<Simplex, V>,
    /// Simplices in filtration order (non-decreasing by value, then dim, then lex).
    order: Vec<Simplex>,
    /// Whether `order` is currently up to date.
    dirty: bool,
}

impl<V: Ord + Clone> Filtration<V> {
    /// Create an empty filtration.
    pub fn new() -> Self {
        Self {
            complex: SimplicialComplex::new(),
            values: HashMap::new(),
            order: vec![],
            dirty: false,
        }
    }

    /// Insert a simplex at `value`.
    ///
    /// All faces of `simplex` that are not yet present are inserted at
    /// `value` as well (faces must enter no later than cofaces).
    ///
    /// If `simplex` is already present with a **smaller** value, the existing
    /// value is kept (faces always enter at the minimum of any coface's value).
    pub fn insert(&mut self, simplex: Simplex, value: V) {
        self.dirty = true;
        self.insert_inner(simplex, value);
    }

    fn insert_inner(&mut self, simplex: Simplex, value: V) {
        // Insert faces first (recursively), using at most `value`.
        for (_sign, face) in simplex.boundary() {
            if let Some(existing) = self.values.get(&face) {
                // Keep the smaller value (face must enter no later than coface).
                if &value < existing {
                    let v = value.clone();
                    self.insert_inner(face, v);
                }
                // else: face already present with a <= value, leave it.
            } else {
                self.insert_inner(face, value.clone());
            }
        }
        // Now insert the simplex itself.
        let entry = self.values.entry(simplex.clone()).or_insert_with(|| {
            self.complex.insert_unchecked(simplex.clone());
            value.clone()
        });
        // If already present but new value is smaller, update.
        if value < *entry {
            *entry = value;
        }
    }

    /// Rebuild the filtration order (sorted by (value, dim, lex)).
    fn rebuild_order(&mut self) {
        if !self.dirty {
            return;
        }
        let mut entries: Vec<(&Simplex, &V)> = self.values.iter().collect();
        entries.sort_by(|(s1, v1), (s2, v2)| {
            v1.cmp(v2)
                .then_with(|| s1.dim().cmp(&s2.dim()))
                .then_with(|| s1.cmp(s2))
        });
        self.order = entries.into_iter().map(|(s, _)| s.clone()).collect();
        self.dirty = false;
    }

    /// The filtration value of `simplex`, or `None` if not in the filtration.
    pub fn value(&self, simplex: &Simplex) -> Option<&V> {
        self.values.get(simplex)
    }

    /// Simplices in filtration order (non-decreasing by value, ties broken by
    /// dimension then lexicographic vertex order).
    pub fn ordered(&mut self) -> &[Simplex] {
        self.rebuild_order();
        &self.order
    }

    /// The underlying simplicial complex (unordered).
    pub fn complex(&self) -> &SimplicialComplex {
        &self.complex
    }

    /// Boundary matrix in column form, with columns in filtration order.
    ///
    /// Returns `Vec<Vec<(i32, usize)>>` where column `j` contains
    /// `(coefficient, row_index)` pairs representing \(\partial \sigma_j\)
    /// in the basis given by the filtration order.
    ///
    /// This is compatible with the column format expected by `lophat` and
    /// `phlite`.  Faces with coefficient 0 are omitted.
    pub fn boundary_matrix(&mut self) -> Vec<Vec<(i32, usize)>> {
        self.rebuild_order();
        // Build index map: simplex -> position in filtration order.
        let index: HashMap<&Simplex, usize> =
            self.order.iter().enumerate().map(|(i, s)| (s, i)).collect();

        self.order
            .iter()
            .map(|sigma| {
                let mut col: Vec<(i32, usize)> = sigma
                    .boundary()
                    .into_iter()
                    .filter_map(|(coeff, face)| index.get(&face).map(|&row| (coeff, row)))
                    .collect();
                // Sort by row index for a canonical column representation.
                col.sort_by_key(|&(_, row)| row);
                col
            })
            .collect()
    }
}

impl<V: Ord + Clone> Default for Filtration<V> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::Simplex;

    fn s(vs: &[usize]) -> Simplex {
        Simplex::new_checked(vs.to_vec()).unwrap()
    }

    // --- OrdF64 tests ---

    #[test]
    fn ord_f64_ordering_matches_f64() {
        let vals: Vec<f64> = vec![-2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 100.0];
        let wrapped: Vec<OrdF64> = vals.iter().map(|&v| OrdF64::new(v)).collect();
        for i in 0..wrapped.len() {
            for j in 0..wrapped.len() {
                let expected = vals[i].partial_cmp(&vals[j]).unwrap();
                let got = wrapped[i].cmp(&wrapped[j]);
                assert_eq!(
                    got, expected,
                    "ordering mismatch at {i},{j}: {} vs {}",
                    vals[i], vals[j]
                );
            }
        }
    }

    #[test]
    fn ord_f64_roundtrip() {
        for v in &[
            -std::f64::consts::PI,
            -1.0,
            -0.0,
            0.0,
            1.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ] {
            if v.is_nan() {
                continue;
            }
            let ord = OrdF64::new(*v);
            let back = ord.value();
            // Use bit comparison for exact roundtrip (handles -0.0 vs 0.0).
            assert_eq!(v.to_bits(), back.to_bits(), "roundtrip failed for {v}");
        }
    }

    #[test]
    #[should_panic]
    fn ord_f64_rejects_nan() {
        OrdF64::new(f64::NAN);
    }

    // --- Filtration tests ---

    #[test]
    fn filtration_empty() {
        let mut f: Filtration<OrdF64> = Filtration::new();
        assert_eq!(f.ordered().len(), 0);
        assert!(f.boundary_matrix().is_empty());
    }

    #[test]
    fn filtration_three_points() {
        // Three isolated vertices at different filtration values.
        let mut f: Filtration<OrdF64> = Filtration::new();
        f.insert(s(&[0]), OrdF64::new(0.0));
        f.insert(s(&[1]), OrdF64::new(1.0));
        f.insert(s(&[2]), OrdF64::new(2.0));
        let ord = f.ordered();
        assert_eq!(ord.len(), 3);
        assert_eq!(ord[0].vertices(), &[0]);
        assert_eq!(ord[1].vertices(), &[1]);
        assert_eq!(ord[2].vertices(), &[2]);
        // All columns empty (no boundaries).
        let bm = f.boundary_matrix();
        assert!(bm.iter().all(|col| col.is_empty()));
    }

    #[test]
    fn filtration_edge_faces_enter_first() {
        // Insert edge [0,1] at time 1.0; faces [0],[1] should appear before it.
        let mut f: Filtration<OrdF64> = Filtration::new();
        f.insert(s(&[0, 1]), OrdF64::new(1.0));
        let ord = f.ordered().to_vec();
        // [0] and [1] at time 1.0, [0,1] at time 1.0 but dim 1 -> after dim 0.
        assert_eq!(ord.len(), 3);
        assert_eq!(ord[0].dim(), 0);
        assert_eq!(ord[1].dim(), 0);
        assert_eq!(ord[2].dim(), 1);
        // Filtration compatibility: faces have value <= coface.
        assert!(f.value(&s(&[0])).unwrap() <= f.value(&s(&[0, 1])).unwrap());
        assert!(f.value(&s(&[1])).unwrap() <= f.value(&s(&[0, 1])).unwrap());
    }

    #[test]
    fn filtration_triangle_ordering_compatibility() {
        // Triangle with vertices at 0.0, edges at 1.0, face at 2.0.
        let mut f: Filtration<OrdF64> = Filtration::new();
        f.insert(s(&[0]), OrdF64::new(0.0));
        f.insert(s(&[1]), OrdF64::new(0.0));
        f.insert(s(&[2]), OrdF64::new(0.0));
        f.insert(s(&[0, 1]), OrdF64::new(1.0));
        f.insert(s(&[0, 2]), OrdF64::new(1.0));
        f.insert(s(&[1, 2]), OrdF64::new(1.0));
        f.insert(s(&[0, 1, 2]), OrdF64::new(2.0));
        let ord = f.ordered().to_vec();
        assert_eq!(ord.len(), 7);
        // All faces of each simplex appear before it.
        let index: HashMap<Simplex, usize> = ord
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, s)| (s, i))
            .collect();
        for (sigma, &j) in &index {
            for (_sign, face) in sigma.boundary() {
                let i = index[&face];
                assert!(
                    i < j,
                    "face {:?} (index {i}) appears after {:?} (index {j})",
                    face.vertices(),
                    sigma.vertices()
                );
            }
        }
    }

    #[test]
    fn boundary_matrix_triangle_explicit() {
        // Build triangle filtration and check boundary matrix structure.
        let mut f: Filtration<OrdF64> = Filtration::new();
        f.insert(s(&[0]), OrdF64::new(0.0));
        f.insert(s(&[1]), OrdF64::new(0.0));
        f.insert(s(&[2]), OrdF64::new(0.0));
        f.insert(s(&[0, 1]), OrdF64::new(1.0));
        f.insert(s(&[0, 2]), OrdF64::new(1.0));
        f.insert(s(&[1, 2]), OrdF64::new(1.0));
        f.insert(s(&[0, 1, 2]), OrdF64::new(2.0));
        let bm = f.boundary_matrix();
        assert_eq!(bm.len(), 7);
        // Vertices have empty boundary columns.
        assert!(bm[0].is_empty());
        assert!(bm[1].is_empty());
        assert!(bm[2].is_empty());
        // Each edge column has exactly 2 entries.
        assert_eq!(bm[3].len(), 2);
        assert_eq!(bm[4].len(), 2);
        assert_eq!(bm[5].len(), 2);
        // The triangle column has exactly 3 entries.
        assert_eq!(bm[6].len(), 3);
        // All row indices are valid (< 7).
        for col in &bm {
            for &(_coeff, row) in col {
                assert!(row < 7, "row index {row} out of bounds");
            }
        }
        // Coefficients are ±1.
        for col in &bm {
            for &(coeff, _row) in col {
                assert!(coeff == 1 || coeff == -1);
            }
        }
    }

    #[test]
    fn filtration_ordering_compatible_with_faces() {
        // For any filtration built correctly, faces always have index < coface.
        // Build a tetrahedron filtration.
        let mut f: Filtration<OrdF64> = Filtration::new();
        f.insert(s(&[0, 1, 2, 3]), OrdF64::new(1.0));
        let ord = f.ordered().to_vec();
        let index: HashMap<Simplex, usize> = ord
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, s)| (s, i))
            .collect();
        for (sigma, &j) in &index {
            for (_sign, face) in sigma.boundary() {
                let i = index[&face];
                assert!(
                    i < j,
                    "face {:?} (index {i}) should precede {:?} (index {j})",
                    face.vertices(),
                    sigma.vertices()
                );
            }
        }
    }
}
