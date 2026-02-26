//! Vietoris-Rips simplicial complex from a 2D point cloud.
//!
//! Generates ~30 points sampled from two clusters, then for a range of
//! epsilon values builds the Vietoris-Rips complex:
//!   - Add an edge (1-simplex) for every pair of points within distance epsilon.
//!   - Add a triangle (2-simplex) for every triple of pairwise-connected points.
//!   - Compute Betti numbers b_0 (connected components) and b_1 (1-cycles)
//!     via rank-nullity on the boundary matrices.
//!
//! As epsilon grows:
//!   - b_0 decreases from n (all isolated) toward 1 (fully connected).
//!   - b_1 may rise then fall: cycles appear before being filled by triangles.
//!
//! Run: cargo run --example vietoris_rips

use skel::topology::Simplex;
use std::collections::{HashMap, HashSet};

// ---------- point cloud generation (deterministic, no rng dep) ----------

/// Two blobs in R^2, centered at (0,0) and (4,0), each with ~15 points
/// arranged on concentric rings. Deterministic (no randomness).
fn two_blobs() -> Vec<[f64; 2]> {
    let mut pts = Vec::new();

    let centers = [[0.0, 0.0], [4.0, 0.0]];
    let radii = [0.3, 0.7, 1.0];
    let counts = [3, 5, 7]; // points per ring

    for center in &centers {
        // center point
        pts.push(*center);
        for (ring, &r) in radii.iter().enumerate() {
            let n = counts[ring];
            for k in 0..n {
                let theta = 2.0 * std::f64::consts::PI * (k as f64) / (n as f64);
                pts.push([center[0] + r * theta.cos(), center[1] + r * theta.sin()]);
            }
        }
    }
    pts
}

fn dist(a: &[f64; 2], b: &[f64; 2]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

// ---------- Rips complex construction ----------

/// Build the 1-skeleton and 2-skeleton of the Vietoris-Rips complex at
/// the given epsilon threshold.
fn rips_complex(
    points: &[[f64; 2]],
    epsilon: f64,
) -> (Vec<Simplex>, Vec<Simplex>, Vec<Simplex>) {
    let n = points.len();

    // 0-simplices
    let vertices: Vec<Simplex> = (0..n)
        .map(|i| Simplex::new_checked(vec![i]).unwrap())
        .collect();

    // Adjacency: which pairs are within epsilon?
    let mut adj: HashSet<(usize, usize)> = HashSet::new();
    let mut edges: Vec<Simplex> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            if dist(&points[i], &points[j]) <= epsilon {
                adj.insert((i, j));
                edges.push(Simplex::new_checked(vec![i, j]).unwrap());
            }
        }
    }

    // 2-simplices: triples where all three pairwise edges exist.
    let mut triangles: Vec<Simplex> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            if !adj.contains(&(i, j)) {
                continue;
            }
            for k in (j + 1)..n {
                if adj.contains(&(i, k)) && adj.contains(&(j, k)) {
                    triangles.push(Simplex::new_checked(vec![i, j, k]).unwrap());
                }
            }
        }
    }

    (vertices, edges, triangles)
}

// ---------- linear algebra (reused pattern from simplicial_complex) ----------

fn boundary_matrix(domain: &[Simplex], codomain: &[Simplex]) -> Vec<Vec<i32>> {
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
        let mut pivot = None;
        for row in r..m {
            if a[row][col].abs() > 1e-10 {
                pivot = Some(row);
                break;
            }
        }
        let Some(pivot_row) = pivot else { continue };
        a.swap(r, pivot_row);
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

/// Compute Betti numbers b_0, b_1 from the chain complex C_2 -> C_1 -> C_0.
fn betti_numbers(
    vertices: &[Simplex],
    edges: &[Simplex],
    triangles: &[Simplex],
) -> (usize, usize) {
    let rank_b1 = if edges.is_empty() {
        0
    } else {
        rank(&boundary_matrix(edges, vertices))
    };

    let rank_b2 = if triangles.is_empty() || edges.is_empty() {
        0
    } else {
        rank(&boundary_matrix(triangles, edges))
    };

    let b0 = vertices.len() - rank_b1;
    let b1 = (edges.len() - rank_b1) - rank_b2;
    (b0, b1)
}

// ---------- main ----------

fn main() {
    let points = two_blobs();
    println!("Vietoris-Rips complex: {} points from two clusters", points.len());
    println!("Cluster centers: (0,0) and (4,0), radii 0.3 / 0.7 / 1.0\n");

    let epsilons = [0.5, 1.0, 1.5, 2.5, 5.0];

    // Header
    println!(
        "{:>9} | {:>10} | {:>13} | {:>4} | {:>4}",
        "epsilon", "num_edges", "num_triangles", "b_0", "b_1"
    );
    println!("{}", "-".repeat(52));

    for &eps in &epsilons {
        let (verts, edges, triangles) = rips_complex(&points, eps);
        let (b0, b1) = betti_numbers(&verts, &edges, &triangles);
        println!(
            "{eps:>9.1} | {edges:>10} | {triangles:>13} | {b0:>4} | {b1:>4}",
            eps = eps,
            edges = edges.len(),
            triangles = triangles.len(),
            b0 = b0,
            b1 = b1,
        );
    }

    println!("\nInterpretation:");
    println!("  - Small epsilon: many components (b_0 high), few or no cycles.");
    println!("  - As epsilon grows, clusters merge (b_0 drops).");
    println!("  - b_1 may spike when edges form cycles before triangles fill them.");
    println!("  - Large epsilon: single component (b_0=1), all cycles filled (b_1=0).");
}
