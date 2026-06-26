//! Persistent homology by reducing a filtration boundary matrix.
//!
//! Builds a small filtered triangle where a 1-cycle appears when the last
//! edge closes the loop, then disappears when the triangle fills it. The
//! example uses `Filtration::boundary_matrix()` as the handoff point and
//! performs the standard left-to-right column reduction over F_2.
//!
//! This is a proof sketch for a future persistence crate boundary, not a
//! general barcode implementation. It is intentionally tiny and explicit.
//!
//! Run: cargo run --example persistence_reduction

use skel::filtration::{Filtration, OrdF64};
use skel::topology::Simplex;
use std::collections::{HashMap, HashSet};

#[derive(Debug)]
struct Interval {
    dim: usize,
    birth: usize,
    death: Option<usize>,
}

fn simplex(vertices: &[usize]) -> Simplex {
    Simplex::new_checked(vertices.to_vec()).unwrap()
}

fn cycle_then_fill() -> Filtration<OrdF64> {
    let mut filtration = Filtration::new();

    for vertex in 0..3 {
        filtration.insert(simplex(&[vertex]), OrdF64::new(0.0));
    }

    filtration.insert(simplex(&[0, 1]), OrdF64::new(1.0));
    filtration.insert(simplex(&[1, 2]), OrdF64::new(1.0));
    filtration.insert(simplex(&[0, 2]), OrdF64::new(2.0));
    filtration.insert(simplex(&[0, 1, 2]), OrdF64::new(3.0));

    filtration
}

fn mod2_column(column: &[(i32, usize)]) -> Vec<usize> {
    let mut rows: Vec<usize> = column
        .iter()
        .filter_map(|(coeff, row)| (coeff.rem_euclid(2) == 1).then_some(*row))
        .collect();
    rows.sort_unstable();
    rows.dedup();
    rows
}

fn xor_sorted(left: &[usize], right: &[usize]) -> Vec<usize> {
    let mut out = Vec::with_capacity(left.len() + right.len());
    let mut i = 0;
    let mut j = 0;

    while i < left.len() || j < right.len() {
        match (left.get(i), right.get(j)) {
            (Some(&a), Some(&b)) if a == b => {
                i += 1;
                j += 1;
            }
            (Some(&a), Some(&b)) if a < b => {
                out.push(a);
                i += 1;
            }
            (Some(_), Some(&b)) => {
                out.push(b);
                j += 1;
            }
            (Some(&a), None) => {
                out.push(a);
                i += 1;
            }
            (None, Some(&b)) => {
                out.push(b);
                j += 1;
            }
            (None, None) => break,
        }
    }

    out
}

fn persistence_intervals(columns: &[Vec<(i32, usize)>], simplices: &[Simplex]) -> Vec<Interval> {
    let mut reduced_columns: Vec<Vec<usize>> = Vec::with_capacity(columns.len());
    let mut low_to_column: HashMap<usize, usize> = HashMap::new();
    let mut killed_births: HashSet<usize> = HashSet::new();
    let mut intervals = Vec::new();

    for (column_index, column) in columns.iter().enumerate() {
        let mut reduced = mod2_column(column);

        while let Some(&low) = reduced.last() {
            let Some(&matching_column) = low_to_column.get(&low) else {
                break;
            };
            reduced = xor_sorted(&reduced, &reduced_columns[matching_column]);
        }

        if let Some(&low) = reduced.last() {
            low_to_column.insert(low, column_index);
            killed_births.insert(low);
            intervals.push(Interval {
                dim: simplices[low].dim(),
                birth: low,
                death: Some(column_index),
            });
        }

        reduced_columns.push(reduced);
    }

    for (column_index, reduced) in reduced_columns.iter().enumerate() {
        if reduced.is_empty() && !killed_births.contains(&column_index) {
            intervals.push(Interval {
                dim: simplices[column_index].dim(),
                birth: column_index,
                death: None,
            });
        }
    }

    intervals.sort_by_key(|interval| (interval.dim, interval.birth, interval.death));
    intervals
}

fn simplex_label(simplex: &Simplex) -> String {
    format!("{:?}", simplex.vertices())
}

fn main() {
    let mut filtration = cycle_then_fill();
    let columns = filtration.boundary_matrix();
    let simplices = filtration.ordered().to_vec();
    let values: Vec<f64> = simplices
        .iter()
        .map(|simplex| filtration.value(simplex).copied().unwrap().value())
        .collect();
    let intervals = persistence_intervals(&columns, &simplices);

    println!("Persistent homology: cycle then fill");
    println!("====================================\n");

    println!("Filtration order:");
    for (index, simplex) in simplices.iter().enumerate() {
        println!(
            "  {index:>2}: t={time:.1} H_{dim} generator {:?}",
            simplex.vertices(),
            time = values[index],
            dim = simplex.dim(),
        );
    }

    println!("\nIntervals:");
    for interval in &intervals {
        let birth = values[interval.birth];
        let birth_simplex = simplex_label(&simplices[interval.birth]);
        match interval.death {
            Some(death_index) => println!(
                "  H_{dim}: [{birth:.1}, {death:.1}) born at {birth_simplex}, killed by {death_simplex}",
                dim = interval.dim,
                death = values[death_index],
                death_simplex = simplex_label(&simplices[death_index]),
            ),
            None => println!(
                "  H_{dim}: [{birth:.1}, inf) born at {birth_simplex}",
                dim = interval.dim,
            ),
        }
    }

    let finite_h0 = intervals
        .iter()
        .filter(|interval| interval.dim == 0 && interval.death.is_some())
        .count();
    let infinite_h0 = intervals
        .iter()
        .filter(|interval| interval.dim == 0 && interval.death.is_none())
        .count();
    let finite_h1: Vec<&Interval> = intervals
        .iter()
        .filter(|interval| interval.dim == 1 && interval.death.is_some())
        .collect();

    assert_eq!(
        finite_h0, 2,
        "two components should merge into the survivor"
    );
    assert_eq!(infinite_h0, 1, "one connected component should persist");
    assert_eq!(
        finite_h1.len(),
        1,
        "one loop should be born and then filled"
    );

    let loop_interval = finite_h1[0];
    assert_eq!(values[loop_interval.birth], 2.0);
    assert_eq!(values[loop_interval.death.unwrap()], 3.0);

    println!("\nCheck: one H_1 class is born at t=2.0 and dies at t=3.0.");
}
