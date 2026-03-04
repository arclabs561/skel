//! Manifold trait: UnitCircle (S^1) implementation.
//!
//! The `Manifold` trait is the core abstraction in skel, consumed by hyperball
//! (Poincare ball, Lorentz model) and flowmatch (Riemannian ODE integration).
//! This example implements the trait for the unit circle -- the simplest
//! non-trivial Riemannian manifold -- and verifies the key identities:
//!
//!   - exp_x(log_x(y)) = y  (round-trip)
//!   - ||log_x(y)|| = geodesic distance d(x, y)
//!   - geodesic distance >= chord distance (metric comparison)
//!
//! Run: cargo run --example manifold_circle

use ndarray::{array, Array1, ArrayView1};
use skel::Manifold;

// ---- UnitCircle: S^1 embedded in R^2 ----

struct UnitCircle;

impl Manifold for UnitCircle {
    /// exp_x(v): walk from x along the circle by arc length ||v||.
    ///
    /// x is a unit vector on S^1; v is a tangent vector (perpendicular to x).
    /// The geodesic is a rotation by angle = ||v||.
    fn exp_map(&self, x: &ArrayView1<f64>, v: &ArrayView1<f64>) -> Array1<f64> {
        let angle = v.dot(v).sqrt();
        if angle < 1e-15 {
            return x.to_owned();
        }
        let c = angle.cos();
        let s = angle.sin();
        let dir = v.mapv(|vi| vi / angle);
        &x.mapv(|xi| xi * c) + &dir.mapv(|di| di * s)
    }

    /// log_x(y): tangent vector at x pointing toward y, with ||v|| = d(x, y).
    fn log_map(&self, x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> Array1<f64> {
        let dot = x.dot(y).clamp(-1.0, 1.0);
        let angle = dot.acos();
        if angle < 1e-15 {
            return Array1::zeros(x.len());
        }
        // Component of y perpendicular to x
        let proj = y - &x.mapv(|xi| xi * dot);
        let norm = proj.dot(&proj).sqrt();
        if norm < 1e-15 {
            return Array1::zeros(x.len());
        }
        proj.mapv(|pi| pi * angle / norm)
    }

    /// Parallel transport from T_x to T_y: rotation by the signed arc angle.
    fn parallel_transport(
        &self,
        x: &ArrayView1<f64>,
        y: &ArrayView1<f64>,
        v: &ArrayView1<f64>,
    ) -> Array1<f64> {
        let log_v = self.log_map(x, y);
        let angle = log_v.dot(&log_v).sqrt();
        if angle < 1e-15 {
            return v.to_owned();
        }
        // Signed angle from x to y (using 2D cross product)
        let cross = x[0] * y[1] - x[1] * y[0];
        let signed = if cross >= 0.0 { angle } else { -angle };
        let cs = signed.cos();
        let sn = signed.sin();
        Array1::from_vec(vec![cs * v[0] - sn * v[1], sn * v[0] + cs * v[1]])
    }

    /// Project back onto S^1 by normalizing.
    fn project(&self, x: &ArrayView1<f64>) -> Array1<f64> {
        let norm = x.dot(x).sqrt();
        x.mapv(|xi| xi / norm)
    }
}

/// Geodesic (arc) distance on S^1.
fn geodesic_distance(x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> f64 {
    let dot = x.dot(y).clamp(-1.0, 1.0);
    dot.acos()
}

/// Chord (straight-line) distance in R^2.
fn chord_distance(x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> f64 {
    let diff = x.to_owned() - y;
    diff.dot(&diff).sqrt()
}

fn main() {
    let circle = UnitCircle;

    println!("Manifold trait: UnitCircle (S^1)");
    println!("================================\n");

    // Two points on the circle
    let x = array![1.0, 0.0]; // angle = 0
    let y = {
        let theta = std::f64::consts::PI / 3.0; // 60 degrees
        array![theta.cos(), theta.sin()]
    };

    println!("x = [{:.4}, {:.4}]  (angle = 0)", x[0], x[1]);
    println!(
        "y = [{:.4}, {:.4}]  (angle = pi/3 = {:.4} rad)\n",
        y[0],
        y[1],
        std::f64::consts::PI / 3.0
    );

    // 1. log_map and exp_map round-trip
    let v = circle.log_map(&x.view(), &y.view());
    let y_recovered = circle.exp_map(&x.view(), &v.view());
    let roundtrip_err = {
        let diff = &y - &y_recovered;
        diff.dot(&diff).sqrt()
    };

    println!("1. exp/log round-trip");
    println!("   log_x(y)         = [{:.6}, {:.6}]", v[0], v[1]);
    println!(
        "   exp_x(log_x(y))  = [{:.6}, {:.6}]",
        y_recovered[0], y_recovered[1]
    );
    println!("   round-trip error  = {:.2e}", roundtrip_err);

    // 2. ||log_x(y)|| = geodesic distance
    let log_norm = v.dot(&v).sqrt();
    let geo_d = geodesic_distance(&x.view(), &y.view());
    println!("\n2. log norm vs geodesic distance");
    println!("   ||log_x(y)|| = {:.6}", log_norm);
    println!("   d_geo(x, y)  = {:.6}", geo_d);
    println!("   difference   = {:.2e}", (log_norm - geo_d).abs());

    // 3. Interpolation along the geodesic
    println!("\n3. Geodesic interpolation from x to y");
    println!(
        "{:>6}  {:>10}  {:>10}  {:>10}",
        "t", "point_x", "point_y", "||p||"
    );
    println!("{:-<6}  {:-<10}  {:-<10}  {:-<10}", "", "", "", "");

    for i in 0..=8 {
        let t = i as f64 / 8.0;
        let v_scaled = v.mapv(|vi| vi * t);
        let p = circle.exp_map(&x.view(), &v_scaled.view());
        let p_norm = p.dot(&p).sqrt();
        println!(
            "{:>6.3}  {:>10.6}  {:>10.6}  {:>10.6}",
            t, p[0], p[1], p_norm
        );
    }
    println!("   (||p|| should be 1.0 at every step -- stays on the circle)");

    // 4. Geodesic vs chord distance at several angles
    println!("\n4. Geodesic vs chord distance");
    println!(
        "{:>10}  {:>12}  {:>12}  {:>8}",
        "angle", "geodesic", "chord", "ratio"
    );
    println!("{:-<10}  {:-<12}  {:-<12}  {:-<8}", "", "", "", "");

    let angles = [
        std::f64::consts::PI / 12.0,
        std::f64::consts::PI / 6.0,
        std::f64::consts::PI / 4.0,
        std::f64::consts::PI / 3.0,
        std::f64::consts::PI / 2.0,
        2.0 * std::f64::consts::PI / 3.0,
        std::f64::consts::PI,
    ];

    for &theta in &angles {
        let q = array![theta.cos(), theta.sin()];
        let g = geodesic_distance(&x.view(), &q.view());
        let c = chord_distance(&x.view(), &q.view());
        println!("{:>10.4}  {:>12.6}  {:>12.6}  {:>7.4}x", theta, g, c, g / c);
    }
    println!("   (geodesic >= chord, equality only at angle = 0)");

    // 5. Parallel transport preserves norm
    println!("\n5. Parallel transport");
    let tangent = array![0.0, 1.0]; // tangent to circle at x = [1,0]
    let transported = circle.parallel_transport(&x.view(), &y.view(), &tangent.view());
    let norm_before = tangent.dot(&tangent).sqrt();
    let norm_after = transported.dot(&transported).sqrt();
    println!(
        "   v at x      = [{:.4}, {:.4}], ||v|| = {:.6}",
        tangent[0], tangent[1], norm_before
    );
    println!(
        "   v at y (PT) = [{:.4}, {:.4}], ||v|| = {:.6}",
        transported[0], transported[1], norm_after
    );
    println!(
        "   norm change = {:.2e}  (should be ~0)",
        (norm_before - norm_after).abs()
    );
}
