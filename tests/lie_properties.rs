//! SO(3) exp/log properties over random axis-angle vectors, plus NaN input.

use proptest::prelude::*;
use skel::lie::{exp_so3, log_so3};

fn det3(r: &[f64; 9]) -> f64 {
    r[0] * (r[4] * r[8] - r[5] * r[7]) - r[1] * (r[3] * r[8] - r[5] * r[6])
        + r[2] * (r[3] * r[7] - r[4] * r[6])
}

proptest! {
    /// For |omega| < pi the logarithm is the unique inverse of the
    /// exponential (Sola et al. 2018, Sec. 4).
    #[test]
    fn so3_log_inverts_exp_inside_pi(
        axis in prop::array::uniform3(-1.0f64..1.0),
        angle in 0.0f64..3.1,
    ) {
        let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        prop_assume!(norm > 1e-3);
        let omega = [axis[0] / norm * angle, axis[1] / norm * angle, axis[2] / norm * angle];
        let back = log_so3(&exp_so3(&omega));
        for i in 0..3 {
            prop_assert!((back[i] - omega[i]).abs() < 1e-7, "omega {omega:?} back {back:?}");
        }
    }

    /// exp maps so(3) into SO(3): R^T R = I and det R = 1.
    #[test]
    fn so3_exp_is_a_rotation(omega in prop::array::uniform3(-10.0f64..10.0)) {
        let r = exp_so3(&omega);
        for i in 0..3 {
            for j in 0..3 {
                let dot: f64 = (0..3).map(|k| r[k * 3 + i] * r[k * 3 + j]).sum();
                let want = if i == j { 1.0 } else { 0.0 };
                prop_assert!((dot - want).abs() < 1e-9);
            }
        }
        prop_assert!((det3(&r) - 1.0).abs() < 1e-9);
    }
}

#[test]
fn so3_log_of_nan_matrix_does_not_panic() {
    let r = [f64::NAN; 9];
    let _ = log_so3(&r);
}
