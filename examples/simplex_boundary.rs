//! Simplicial complex basics: build simplices and compute boundaries.
//!
//! Demonstrates the chain complex identity: the boundary of a boundary is zero (dd = 0).
//! This is the fundamental theorem of algebraic topology that makes homology well-defined.
//!
//! Run: cargo run --example simplex_boundary

use skel::topology::Simplex;
use std::collections::HashMap;

fn main() {
    println!("=== Simplex boundary computation ===\n");

    // A 0-simplex (vertex) has empty boundary
    let v = Simplex::new_canonical(vec![0]).unwrap();
    println!("0-simplex [0]: boundary = {:?}", v.boundary());

    // A 1-simplex (edge) has two oriented boundary vertices
    let edge = Simplex::new_canonical(vec![0, 1]).unwrap();
    let bd = edge.boundary();
    println!("1-simplex [0,1]: boundary = {:?}", bd);
    println!("  (oriented: +[1] - [0], i.e., target minus source)\n");

    // A 2-simplex (triangle) has three oriented boundary edges
    let tri = Simplex::new_canonical(vec![0, 1, 2]).unwrap();
    let bd = tri.boundary();
    println!("2-simplex [0,1,2]: boundary =");
    for (sign, face) in &bd {
        println!("  {:+} * {:?}", sign, face.vertices());
    }

    // Verify dd = 0 (chain complex identity)
    println!("\nChain complex check: d(d([0,1,2])) should cancel to zero:");
    let mut vertex_counts: HashMap<Vec<usize>, i32> = HashMap::new();
    for (sign, face) in &bd {
        let sub_bd = face.boundary();
        for (sub_sign, vertex) in &sub_bd {
            let total_sign = sign * sub_sign;
            *vertex_counts.entry(vertex.vertices().to_vec()).or_default() += total_sign;
        }
    }
    let all_zero = vertex_counts.values().all(|&v| v == 0);
    println!("  vertex coefficients: {:?}", vertex_counts);
    println!("  dd = 0? {}\n", all_zero);

    // A 3-simplex (tetrahedron): 4 boundary faces
    let tet = Simplex::new_canonical(vec![0, 1, 2, 3]).unwrap();
    let bd = tet.boundary();
    println!(
        "3-simplex [0,1,2,3] (tetrahedron): {} boundary faces",
        bd.len()
    );
    for (sign, face) in &bd {
        println!("  {:+} * {:?}", sign, face.vertices());
    }

    // Verify dd = 0 for tetrahedron too
    let mut edge_counts: HashMap<Vec<usize>, i32> = HashMap::new();
    for (sign, face) in &bd {
        let sub_bd = face.boundary();
        for (sub_sign, edge) in &sub_bd {
            let total_sign = sign * sub_sign;
            *edge_counts.entry(edge.vertices().to_vec()).or_default() += total_sign;
        }
    }
    let all_zero = edge_counts.values().all(|&v| v == 0);
    println!("  dd = 0? {}", all_zero);

    println!("\nThe dd = 0 identity makes homology well-defined:");
    println!("  cycles (ker d) / boundaries (im d) = homology groups");
}
