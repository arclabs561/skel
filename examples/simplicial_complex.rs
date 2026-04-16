#![allow(clippy::needless_range_loop)]
//! Simplicial complex: boundary matrices, dd=0, and Betti numbers.
//!
//! Builds the surface of a tetrahedron (4 triangles, 6 edges, 4 vertices),
//! constructs the boundary matrices B_2 (triangles -> edges) and B_1
//! (edges -> vertices), verifies B_1 * B_2 = 0, and computes Betti numbers
//! by rank-nullity.
//!
//! This extends simplex_boundary.rs by showing the *consequence* of dd=0:
//! the chain complex structure that makes homology well-defined.
//!
//! The tetrahedron surface is homeomorphic to S^2 (a sphere), so:
//!   b_0 = 1 (connected)
//!   b_1 = 0 (no 1-cycles that aren't boundaries)
//!   b_2 = 1 (the whole surface is a 2-cycle)
//!
//! Run: cargo run --example simplicial_complex

use skel::topology::Simplex;
use std::collections::HashMap;

/// Build the simplicial complex for the surface of a tetrahedron.
/// Vertices: {0, 1, 2, 3}.
/// Triangles (2-simplices): all 4 faces of the tetrahedron.
fn tetrahedron_surface() -> (Vec<Simplex>, Vec<Simplex>, Vec<Simplex>) {
    // 0-simplices (vertices)
    let vertices: Vec<Simplex> = (0..4)
        .map(|i| Simplex::new_checked(vec![i]).unwrap())
        .collect();

    // 1-simplices (edges): all (4 choose 2) = 6 edges
    let edges: Vec<Simplex> = vec![
        Simplex::new_checked(vec![0, 1]).unwrap(),
        Simplex::new_checked(vec![0, 2]).unwrap(),
        Simplex::new_checked(vec![0, 3]).unwrap(),
        Simplex::new_checked(vec![1, 2]).unwrap(),
        Simplex::new_checked(vec![1, 3]).unwrap(),
        Simplex::new_checked(vec![2, 3]).unwrap(),
    ];

    // 2-simplices (triangles): all 4 faces of the tetrahedron
    let triangles: Vec<Simplex> = vec![
        Simplex::new_checked(vec![0, 1, 2]).unwrap(),
        Simplex::new_checked(vec![0, 1, 3]).unwrap(),
        Simplex::new_checked(vec![0, 2, 3]).unwrap(),
        Simplex::new_checked(vec![1, 2, 3]).unwrap(),
    ];

    (vertices, edges, triangles)
}

/// Build the boundary matrix B_k as a dense i32 matrix (rows x cols),
/// mapping k-simplices (columns) to (k-1)-simplices (rows).
fn boundary_matrix(
    domain: &[Simplex],   // k-simplices (columns)
    codomain: &[Simplex], // (k-1)-simplices (rows)
) -> Vec<Vec<i32>> {
    // Map each codomain simplex to its row index
    let row_index: HashMap<Vec<usize>, usize> = codomain
        .iter()
        .enumerate()
        .map(|(i, s)| (s.vertices().to_vec(), i))
        .collect();

    let nrows = codomain.len();
    let ncols = domain.len();
    let mut mat = vec![vec![0i32; ncols]; nrows];

    for (col, sigma) in domain.iter().enumerate() {
        for (sign, face) in sigma.boundary() {
            if let Some(&row) = row_index.get(face.vertices()) {
                mat[row][col] = sign;
            }
        }
    }

    mat
}

/// Multiply two dense i32 matrices.
fn mat_mul(a: &[Vec<i32>], b: &[Vec<i32>]) -> Vec<Vec<i32>> {
    let m = a.len();
    let n = b[0].len();
    let k = b.len();
    assert_eq!(a[0].len(), k, "dimension mismatch in matrix multiply");

    let mut c = vec![vec![0i32; n]; m];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0i32;
            for p in 0..k {
                s += a[i][p] * b[p][j];
            }
            c[i][j] = s;
        }
    }
    c
}

/// Check if a matrix is all zeros.
fn is_zero_matrix(mat: &[Vec<i32>]) -> bool {
    mat.iter().all(|row| row.iter().all(|&v| v == 0))
}

/// Compute the rank of an integer matrix using Gaussian elimination over Q.
/// (Works on a copy converted to f64.)
fn rank(mat: &[Vec<i32>]) -> usize {
    if mat.is_empty() || mat[0].is_empty() {
        return 0;
    }
    let m = mat.len();
    let n = mat[0].len();
    let mut a: Vec<Vec<f64>> = mat
        .iter()
        .map(|row| row.iter().map(|&v| v as f64).collect())
        .collect();

    let mut r = 0;
    for col in 0..n {
        // Find pivot
        let mut pivot = None;
        for row in r..m {
            if a[row][col].abs() > 1e-10 {
                pivot = Some(row);
                break;
            }
        }
        let Some(pivot_row) = pivot else {
            continue;
        };

        // Swap pivot row to position r
        a.swap(r, pivot_row);

        // Eliminate below
        let scale = a[r][col];
        for row in 0..m {
            if row == r {
                continue;
            }
            let factor = a[row][col] / scale;
            for c in col..n {
                a[row][c] -= factor * a[r][c];
            }
        }
        r += 1;
    }
    r
}

fn print_matrix(name: &str, mat: &[Vec<i32>], row_labels: &[String], col_labels: &[String]) {
    println!("{}  ({}x{}):", name, mat.len(), mat[0].len());

    // Column headers
    print!("  {:>12}", "");
    for label in col_labels {
        print!(" {:>6}", label);
    }
    println!();

    for (i, row) in mat.iter().enumerate() {
        print!("  {:>12}", row_labels[i]);
        for &v in row {
            print!(" {:>6}", v);
        }
        println!();
    }
    println!();
}

fn simplex_label(s: &Simplex) -> String {
    format!("{:?}", s.vertices())
}

fn main() {
    println!("Simplicial complex: tetrahedron surface");
    println!("=======================================\n");

    let (vertices, edges, triangles) = tetrahedron_surface();

    println!(
        "Complex: {} vertices, {} edges, {} triangles\n",
        vertices.len(),
        edges.len(),
        triangles.len()
    );

    // Build boundary matrices
    let b1 = boundary_matrix(&edges, &vertices);
    let b2 = boundary_matrix(&triangles, &edges);

    let vert_labels: Vec<String> = vertices.iter().map(simplex_label).collect();
    let edge_labels: Vec<String> = edges.iter().map(simplex_label).collect();
    let tri_labels: Vec<String> = triangles.iter().map(simplex_label).collect();

    // Print B_1 (edges -> vertices)
    print_matrix("B_1 (edges -> vertices)", &b1, &vert_labels, &edge_labels);

    // Print B_2 (triangles -> edges)
    print_matrix("B_2 (triangles -> edges)", &b2, &edge_labels, &tri_labels);

    // Verify dd = 0: B_1 * B_2 should be the zero matrix
    let product = mat_mul(&b1, &b2);
    let dd_zero = is_zero_matrix(&product);
    println!("Chain complex identity: B_1 * B_2 = 0?  {}\n", dd_zero);

    if !dd_zero {
        println!("  ERROR: B_1 * B_2 is not zero. The boundary matrices are inconsistent.");
        print_matrix("B_1 * B_2", &product, &vert_labels, &tri_labels);
        return;
    }

    // Compute Betti numbers via rank-nullity.
    //
    // For the chain complex  C_2 --B_2--> C_1 --B_1--> C_0:
    //
    //   b_k = dim(ker B_k) - dim(im B_{k+1})
    //       = (dim C_k - rank B_k) - rank B_{k+1}
    //
    // With B_0 = 0 (empty) and B_3 = 0 (no 3-simplices in the surface).

    let rank_b1 = rank(&b1);
    let rank_b2 = rank(&b2);

    let dim_c0 = vertices.len();
    let dim_c1 = edges.len();
    let dim_c2 = triangles.len();

    // b_0 = nullity(B_0) - rank(B_1) = dim_c0 - rank_b1
    // (B_0 is the zero map from C_0 to nothing, so nullity(B_0) = dim_c0)
    let b0 = dim_c0 - rank_b1;

    // b_1 = nullity(B_1) - rank(B_2) = (dim_c1 - rank_b1) - rank_b2
    let b1_betti = (dim_c1 - rank_b1) - rank_b2;

    // b_2 = nullity(B_2) - rank(B_3) = (dim_c2 - rank_b2) - 0
    let b2_betti = dim_c2 - rank_b2;

    println!("Ranks:");
    println!(
        "  rank(B_1) = {}  (of {}x{} matrix)",
        rank_b1, dim_c0, dim_c1
    );
    println!(
        "  rank(B_2) = {}  (of {}x{} matrix)",
        rank_b2, dim_c1, dim_c2
    );

    println!("\nBetti numbers (by rank-nullity):");
    println!("  b_0 = {}  (connected components)", b0);
    println!("  b_1 = {}  (independent 1-cycles)", b1_betti);
    println!("  b_2 = {}  (independent 2-cycles)", b2_betti);

    println!("\nExpected for a sphere (S^2): b_0=1, b_1=0, b_2=1");
    println!(
        "Euler characteristic: b_0 - b_1 + b_2 = {} (should be 2 for S^2)",
        b0 as i32 - b1_betti as i32 + b2_betti as i32
    );
}
