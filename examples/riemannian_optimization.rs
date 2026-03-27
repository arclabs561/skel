//! Riemannian optimization on the unit sphere S^2.
//!
//! Demonstrates Riemannian SGD finding the point on S^2 closest to a target,
//! printing geodesic distance at each step to show convergence.

use ndarray::{array, Array1, ArrayView1};
use skel::optim::{
    geodesic_distance, riemannian_adam_step, riemannian_sgd_step, RiemannianAdamState,
};
use skel::Manifold;

/// Unit sphere in R^n.
struct Sphere;

impl Manifold for Sphere {
    fn exp_map(&self, x: &ArrayView1<f64>, v: &ArrayView1<f64>) -> Array1<f64> {
        let norm_v = v.iter().map(|vi| vi * vi).sum::<f64>().sqrt();
        if norm_v < 1e-15 {
            return x.to_owned();
        }
        let cos_t = norm_v.cos();
        let sin_t = norm_v.sin();
        let mut result = Array1::zeros(x.len());
        for i in 0..x.len() {
            result[i] = cos_t * x[i] + sin_t * (v[i] / norm_v);
        }
        result
    }

    fn log_map(&self, x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> Array1<f64> {
        let dot: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let dot = dot.clamp(-1.0, 1.0);
        let theta = dot.acos();
        if theta < 1e-15 {
            return Array1::zeros(x.len());
        }
        let mut u = Array1::zeros(x.len());
        for i in 0..x.len() {
            u[i] = y[i] - dot * x[i];
        }
        let norm_u = u.iter().map(|ui| ui * ui).sum::<f64>().sqrt();
        if norm_u < 1e-15 {
            return Array1::zeros(x.len());
        }
        u.mapv(|ui| ui * theta / norm_u)
    }

    fn parallel_transport(
        &self,
        x: &ArrayView1<f64>,
        y: &ArrayView1<f64>,
        v: &ArrayView1<f64>,
    ) -> Array1<f64> {
        let log_xy = self.log_map(x, y);
        let dot_xy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let denom = 1.0 + dot_xy;
        if denom.abs() < 1e-15 {
            return v.to_owned();
        }
        let dot_v_log: f64 = v.iter().zip(log_xy.iter()).map(|(a, b)| a * b).sum();
        let coeff = dot_v_log / denom;
        let mut result = Array1::zeros(v.len());
        for i in 0..v.len() {
            result[i] = v[i] - coeff * (x[i] + y[i]);
        }
        result
    }

    fn project(&self, x: &ArrayView1<f64>) -> Array1<f64> {
        let norm = x.iter().map(|xi| xi * xi).sum::<f64>().sqrt();
        if norm < 1e-15 {
            let mut result = Array1::zeros(x.len());
            if !result.is_empty() {
                result[0] = 1.0;
            }
            return result;
        }
        x.mapv(|xi| xi / norm)
    }
}

fn main() {
    let manifold = Sphere;
    let target = array![0.0, 0.0, 1.0]; // north pole
    let start = array![1.0, 0.0, 0.0]; // on the equator

    let lr = 0.1;
    let steps = 30;

    // --- Riemannian SGD ---
    println!("Riemannian SGD on S^2: minimizing distance to north pole");
    println!("{:<6} {:<12} {:<20}", "step", "distance", "point");
    let mut x = start.clone();
    for step in 0..=steps {
        let dist = geodesic_distance(&manifold, &x.view(), &target.view());
        if step % 5 == 0 {
            println!(
                "{:<6} {:<12.6} [{:.4}, {:.4}, {:.4}]",
                step, dist, x[0], x[1], x[2]
            );
        }
        if step < steps {
            // Gradient of 0.5 * d(x, target)^2 is -log_x(target).
            let log = manifold.log_map(&x.view(), &target.view());
            let grad: Array1<f64> = log.mapv(|v| -v);
            x = riemannian_sgd_step(&manifold, &x.view(), &grad.view(), lr);
        }
    }

    // --- Riemannian Adam ---
    println!("\nRiemannian Adam on S^2: minimizing distance to north pole");
    println!("{:<6} {:<12} {:<20}", "step", "distance", "point");
    let mut x = start.clone();
    let mut state = RiemannianAdamState::new(start);
    for step in 0..=steps {
        let dist = geodesic_distance(&manifold, &x.view(), &target.view());
        if step % 5 == 0 {
            println!(
                "{:<6} {:<12.6} [{:.4}, {:.4}, {:.4}]",
                step, dist, x[0], x[1], x[2]
            );
        }
        if step < steps {
            let log = manifold.log_map(&x.view(), &target.view());
            let grad: Array1<f64> = log.mapv(|v| -v);
            x = riemannian_adam_step(
                &manifold,
                &x.view(),
                &grad.view(),
                &mut state,
                lr,
                0.9,
                0.999,
                1e-8,
            );
        }
    }
}
