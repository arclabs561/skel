//! Lie group primitives: SO(3) and SE(3).
//!
//! Lie groups combine group structure (composition, inversion) with smooth
//! manifold structure.  The exponential and logarithmic maps connect the
//! **Lie algebra** (tangent space at the identity) to the group itself,
//! enabling geodesic interpolation and flow matching on rotation and
//! rigid-body transformation spaces.
//!
//! # SO(3) -- the rotation group
//!
//! Elements are 3x3 orthogonal matrices with determinant +1, stored as
//! `[f64; 9]` in row-major order.  The Lie algebra `so(3)` is the space
//! of 3x3 skew-symmetric matrices, parameterized by axis-angle vectors
//! in `[f64; 3]`.
//!
//! The exponential map uses the **Rodrigues formula**:
//!
//! $$
//! \exp(\omega) = I + \frac{\sin\theta}{\theta}[\omega]_\times
//!     + \frac{1 - \cos\theta}{\theta^2}[\omega]_\times^2
//! $$
//!
//! where `theta = ||omega||` and `[omega]_x` is the skew-symmetric matrix
//! of `omega`.
//!
//! # SE(3) -- the special Euclidean group
//!
//! Rigid body transformations: rotation + translation.  Represented as
//! `([f64; 9], [f64; 3])` -- a rotation matrix and a translation vector.
//! The Lie algebra `se(3)` is parameterized by 6D twist vectors
//! `[f64; 6]` where the first 3 components are angular velocity and the
//! last 3 are linear velocity.
//!
//! # References
//!
//! - Sherry & Smets (2025), "Flow Matching on Lie Groups" -- geodesic
//!   interpolation on SO(3) and SE(3) for generative modeling.
//! - Sola et al. (2018), "A micro Lie theory for state estimation in
//!   robotics" -- compact reference for exp/log on SO(3) and SE(3).

// ---------------------------------------------------------------------------
// SO(3) -- rotation group
// ---------------------------------------------------------------------------

/// 3x3 rotation matrix stored row-major as `[f64; 9]`.
pub type RotationMatrix = [f64; 9];

/// 3D axis-angle vector (element of the Lie algebra `so(3)`).
pub type AxisAngle = [f64; 3];

/// The 3x3 identity matrix (row-major).
pub const IDENTITY_SO3: RotationMatrix = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

/// The identity element of SE(3): no rotation, no translation.
pub const IDENTITY_SE3: (RotationMatrix, [f64; 3]) = (IDENTITY_SO3, [0.0, 0.0, 0.0]);

/// Threshold below which we use Taylor expansions to avoid division by zero.
const SMALL_ANGLE: f64 = 1e-10;

/// Exponential map on SO(3): axis-angle vector -> rotation matrix.
///
/// Implements the Rodrigues formula.  For small angles (`||omega|| < 1e-10`),
/// uses a first-order Taylor expansion for numerical stability.
pub fn exp_so3(omega: &AxisAngle) -> RotationMatrix {
    let theta_sq = omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2];
    let theta = theta_sq.sqrt();

    // Coefficients: sin(theta)/theta and (1 - cos(theta))/theta^2.
    let (a, b) = if theta < SMALL_ANGLE {
        // Taylor: sin(t)/t ~ 1 - t^2/6, (1-cos(t))/t^2 ~ 1/2 - t^2/24
        (1.0 - theta_sq / 6.0, 0.5 - theta_sq / 24.0)
    } else {
        (theta.sin() / theta, (1.0 - theta.cos()) / theta_sq)
    };

    // Skew-symmetric matrix [omega]_x components.
    let (wx, wy, wz) = (omega[0], omega[1], omega[2]);

    // [omega]_x:
    //   0   -wz   wy
    //  wz    0   -wx
    // -wy   wx    0
    //
    // [omega]_x^2:
    //  -(wy^2+wz^2)    wx*wy          wx*wz
    //   wx*wy         -(wx^2+wz^2)    wy*wz
    //   wx*wz          wy*wz         -(wx^2+wy^2)

    let k00 = -(wy * wy + wz * wz);
    let k01 = wx * wy;
    let k02 = wx * wz;
    let k11 = -(wx * wx + wz * wz);
    let k12 = wy * wz;
    let k22 = -(wx * wx + wy * wy);

    [
        1.0 + b * k00,
        -a * wz + b * k01,
        a * wy + b * k02,
        a * wz + b * k01,
        1.0 + b * k11,
        -a * wx + b * k12,
        -a * wy + b * k02,
        a * wx + b * k12,
        1.0 + b * k22,
    ]
}

/// Logarithmic map on SO(3): rotation matrix -> axis-angle vector.
///
/// Inverts the exponential map.  Returns the axis-angle vector `omega`
/// such that `exp_so3(omega) = R`.  The angle `||omega||` is in `[0, pi]`.
///
/// For rotations near the identity (`theta ~ 0`), uses a Taylor expansion.
/// For rotations near `pi`, extracts the axis from the symmetric part of `R`.
pub fn log_so3(r: &RotationMatrix) -> AxisAngle {
    // cos(theta) = (trace(R) - 1) / 2
    let cos_theta = ((r[0] + r[4] + r[8]) - 1.0) / 2.0;
    let cos_theta = cos_theta.clamp(-1.0, 1.0);
    let theta = cos_theta.acos();

    if theta < SMALL_ANGLE {
        // Near identity: R ~ I + theta * [omega_hat]_x
        // Extract skew part directly (first-order).
        let factor = 0.5 + theta * theta / 12.0; // Taylor of theta/(2*sin(theta))
        return [
            factor * (r[7] - r[5]), // (R[2,1] - R[1,2]) / 2
            factor * (r[2] - r[6]), // (R[0,2] - R[2,0]) / 2
            factor * (r[3] - r[1]), // (R[1,0] - R[0,1]) / 2
        ];
    }

    if (std::f64::consts::PI - theta) < SMALL_ANGLE {
        // Near pi: sin(theta) ~ 0, use symmetric part.
        // R + R^T = 2 * cos(theta) * I + 2 * (1 - cos(theta)) * (n n^T)
        // where n is the rotation axis.
        let sym = [r[0] + 1.0, r[4] + 1.0, r[8] + 1.0]; // diagonal of R + I
        // Pick the largest diagonal element for numerical stability.
        let (idx, &max_val) = sym
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        let scale = (max_val / 2.0).max(0.0).sqrt();
        let mut axis = [0.0; 3];
        axis[idx] = scale;
        // Off-diagonal: R[i,j] + R[j,i] = 2*(1 - cos(theta)) * n_i * n_j
        let denom = 2.0 * scale;
        if denom > SMALL_ANGLE {
            for j in 0..3 {
                if j != idx {
                    // R[idx,j] + R[j,idx]
                    let r_ij = r[idx * 3 + j];
                    let r_ji = r[j * 3 + idx];
                    axis[j] = (r_ij + r_ji) / (2.0 * denom);
                }
            }
        }
        return [
            axis[0] * theta,
            axis[1] * theta,
            axis[2] * theta,
        ];
    }

    // General case: theta / (2 * sin(theta)) * skew(R)
    let factor = theta / (2.0 * theta.sin());
    [
        factor * (r[7] - r[5]),
        factor * (r[2] - r[6]),
        factor * (r[3] - r[1]),
    ]
}

/// Geodesic (slerp) interpolation between two rotations on SO(3).
///
/// Returns the rotation at fraction `t` along the geodesic from `r0` to `r1`.
/// At `t = 0` returns `r0`; at `t = 1` returns `r1`.
///
/// Computed as `r0 * exp(t * log(r0^T * r1))`.
pub fn geodesic_interpolation_so3(r0: &RotationMatrix, r1: &RotationMatrix, t: f64) -> RotationMatrix {
    // delta = r0^T * r1
    let r0t = transpose3(r0);
    let delta = mat_mul3(&r0t, r1);
    let omega = log_so3(&delta);
    let scaled = [omega[0] * t, omega[1] * t, omega[2] * t];
    let r_t = exp_so3(&scaled);
    mat_mul3(r0, &r_t)
}

// ---------------------------------------------------------------------------
// SE(3) -- rigid body transformations
// ---------------------------------------------------------------------------

/// Twist vector: 6D element of the Lie algebra `se(3)`.
///
/// Layout: `[omega_x, omega_y, omega_z, v_x, v_y, v_z]` where
/// `omega` is angular velocity and `v` is linear velocity.
pub type Twist = [f64; 6];

/// Rigid body transformation: `(rotation, translation)`.
pub type RigidTransform = (RotationMatrix, [f64; 3]);

/// Exponential map on SE(3): twist -> rigid body transformation.
///
/// Given a twist `xi = (omega, v)`, computes the rotation `R = exp(omega)`
/// and translation `t = V * v`, where `V` is the left Jacobian of SO(3):
///
/// $$
/// V = I + \frac{1 - \cos\theta}{\theta^2}[\omega]_\times
///     + \frac{\theta - \sin\theta}{\theta^3}[\omega]_\times^2
/// $$
pub fn exp_se3(xi: &Twist) -> RigidTransform {
    let omega = [xi[0], xi[1], xi[2]];
    let v = [xi[3], xi[4], xi[5]];

    let theta_sq = omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2];
    let theta = theta_sq.sqrt();

    let rot = exp_so3(&omega);

    // Compute V * v (left Jacobian applied to linear velocity).
    let (a, b) = if theta < SMALL_ANGLE {
        // Taylor: (1-cos)/t^2 ~ 1/2, (t-sin)/t^3 ~ 1/6
        (0.5 - theta_sq / 24.0, 1.0 / 6.0 - theta_sq / 120.0)
    } else {
        (
            (1.0 - theta.cos()) / theta_sq,
            (theta - theta.sin()) / (theta_sq * theta),
        )
    };

    let (wx, wy, wz) = (omega[0], omega[1], omega[2]);

    // V = I + a * [omega]_x + b * [omega]_x^2
    // Compute V * v directly without forming V explicitly.
    // [omega]_x * v = omega x v (cross product)
    let cross = [
        wy * v[2] - wz * v[1],
        wz * v[0] - wx * v[2],
        wx * v[1] - wy * v[0],
    ];
    // [omega]_x^2 * v = omega x (omega x v)
    let cross2 = [
        wy * cross[2] - wz * cross[1],
        wz * cross[0] - wx * cross[2],
        wx * cross[1] - wy * cross[0],
    ];

    let translation = [
        v[0] + a * cross[0] + b * cross2[0],
        v[1] + a * cross[1] + b * cross2[1],
        v[2] + a * cross[2] + b * cross2[2],
    ];

    (rot, translation)
}

/// Logarithmic map on SE(3): rigid body transformation -> twist.
///
/// Inverts [`exp_se3`].  Recovers the twist vector `(omega, v)` such that
/// `exp_se3((omega, v)) = (R, t)`.
pub fn log_se3(transform: &RigidTransform) -> Twist {
    let (rot, trans) = transform;
    let omega = log_so3(rot);
    let theta_sq = omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2];
    let theta = theta_sq.sqrt();

    // Compute V^{-1} * t.
    // V^{-1} = I - 0.5 * [omega]_x + c * [omega]_x^2
    // where c = (1/theta^2)(1 - theta*sin(theta) / (2*(1 - cos(theta))))
    let (wx, wy, wz) = (omega[0], omega[1], omega[2]);

    let c = if theta < SMALL_ANGLE {
        // Taylor: c ~ 1/12 + theta^2/720
        1.0 / 12.0 + theta_sq / 720.0
    } else {
        let half_theta = theta / 2.0;
        (1.0 / theta_sq) * (1.0 - half_theta * theta.sin() / (1.0 - theta.cos()))
    };

    // [omega]_x * t
    let cross = [
        wy * trans[2] - wz * trans[1],
        wz * trans[0] - wx * trans[2],
        wx * trans[1] - wy * trans[0],
    ];
    // [omega]_x^2 * t
    let cross2 = [
        wy * cross[2] - wz * cross[1],
        wz * cross[0] - wx * cross[2],
        wx * cross[1] - wy * cross[0],
    ];

    let v = [
        trans[0] - 0.5 * cross[0] + c * cross2[0],
        trans[1] - 0.5 * cross[1] + c * cross2[1],
        trans[2] - 0.5 * cross[2] + c * cross2[2],
    ];

    [omega[0], omega[1], omega[2], v[0], v[1], v[2]]
}

/// Geodesic interpolation between two rigid body transforms on SE(3).
///
/// Returns the transform at fraction `t` along the geodesic from `p0` to `p1`.
/// At `t = 0` returns `p0`; at `t = 1` returns `p1`.
///
/// Computed as `p0 * exp(t * log(p0^{-1} * p1))`.
pub fn geodesic_interpolation_se3(
    p0: &RigidTransform,
    p1: &RigidTransform,
    t: f64,
) -> RigidTransform {
    let p0_inv = inverse_se3(p0);
    let delta = compose_se3(&p0_inv, p1);
    let xi = log_se3(&delta);
    let scaled = [
        xi[0] * t,
        xi[1] * t,
        xi[2] * t,
        xi[3] * t,
        xi[4] * t,
        xi[5] * t,
    ];
    let step = exp_se3(&scaled);
    compose_se3(p0, &step)
}

/// Compose two SE(3) transforms: `(R1, t1) * (R2, t2) = (R1*R2, R1*t2 + t1)`.
pub fn compose_se3(a: &RigidTransform, b: &RigidTransform) -> RigidTransform {
    let rot = mat_mul3(&a.0, &b.0);
    let t = [
        a.0[0] * b.1[0] + a.0[1] * b.1[1] + a.0[2] * b.1[2] + a.1[0],
        a.0[3] * b.1[0] + a.0[4] * b.1[1] + a.0[5] * b.1[2] + a.1[1],
        a.0[6] * b.1[0] + a.0[7] * b.1[1] + a.0[8] * b.1[2] + a.1[2],
    ];
    (rot, t)
}

/// Inverse of an SE(3) transform: `(R, t)^{-1} = (R^T, -R^T * t)`.
pub fn inverse_se3(p: &RigidTransform) -> RigidTransform {
    let rt = transpose3(&p.0);
    let t = [
        -(rt[0] * p.1[0] + rt[1] * p.1[1] + rt[2] * p.1[2]),
        -(rt[3] * p.1[0] + rt[4] * p.1[1] + rt[5] * p.1[2]),
        -(rt[6] * p.1[0] + rt[7] * p.1[1] + rt[8] * p.1[2]),
    ];
    (rt, t)
}

// ---------------------------------------------------------------------------
// 3x3 matrix helpers (row-major)
// ---------------------------------------------------------------------------

/// Multiply two 3x3 matrices stored row-major.
fn mat_mul3(a: &[f64; 9], b: &[f64; 9]) -> [f64; 9] {
    [
        a[0] * b[0] + a[1] * b[3] + a[2] * b[6],
        a[0] * b[1] + a[1] * b[4] + a[2] * b[7],
        a[0] * b[2] + a[1] * b[5] + a[2] * b[8],
        a[3] * b[0] + a[4] * b[3] + a[5] * b[6],
        a[3] * b[1] + a[4] * b[4] + a[5] * b[7],
        a[3] * b[2] + a[4] * b[5] + a[5] * b[8],
        a[6] * b[0] + a[7] * b[3] + a[8] * b[6],
        a[6] * b[1] + a[7] * b[4] + a[8] * b[7],
        a[6] * b[2] + a[7] * b[5] + a[8] * b[8],
    ]
}

/// Transpose a 3x3 matrix stored row-major.
fn transpose3(m: &[f64; 9]) -> [f64; 9] {
    [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]]
}

/// Frobenius norm of the difference between two 3x3 matrices.
#[cfg(test)]
fn mat_dist(a: &[f64; 9], b: &[f64; 9]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const TOL: f64 = 1e-10;

    // -- SO(3) tests -------------------------------------------------------

    #[test]
    fn exp_log_roundtrip_so3() {
        // Various axis-angle vectors covering different regimes.
        let cases: &[AxisAngle] = &[
            [0.1, 0.2, 0.3],         // small angle
            [1.0, 0.0, 0.0],         // 1 radian about x
            [0.0, 2.5, 0.0],         // large angle about y
            [0.0, 0.0, 0.1],         // small about z
            [0.5, -0.3, 0.8],        // general
        ];

        for omega in cases {
            let r = exp_so3(omega);
            let recovered = log_so3(&r);
            let r2 = exp_so3(&recovered);
            assert!(
                mat_dist(&r, &r2) < TOL,
                "exp(log(R)) != R for omega = {omega:?}, dist = {}",
                mat_dist(&r, &r2)
            );
        }
    }

    #[test]
    fn identity_maps_to_zero() {
        let omega = log_so3(&IDENTITY_SO3);
        let norm = (omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2]).sqrt();
        assert!(norm < TOL, "log(I) should be zero, got {omega:?} (norm = {norm})");
    }

    #[test]
    fn zero_maps_to_identity() {
        let r = exp_so3(&[0.0, 0.0, 0.0]);
        assert!(
            mat_dist(&r, &IDENTITY_SO3) < TOL,
            "exp(0) should be identity, got {r:?}"
        );
    }

    #[test]
    fn rotation_matrix_is_orthogonal() {
        let cases: &[AxisAngle] = &[
            [0.3, 0.5, -0.2],
            [PI * 0.99, 0.0, 0.0], // near pi
            [0.0, 0.0, PI / 4.0],
            [1.0, 1.0, 1.0],
        ];

        for omega in cases {
            let r = exp_so3(omega);
            let rt = transpose3(&r);
            let product = mat_mul3(&r, &rt);
            assert!(
                mat_dist(&product, &IDENTITY_SO3) < 1e-9,
                "R * R^T != I for omega = {omega:?}, dist = {}",
                mat_dist(&product, &IDENTITY_SO3)
            );

            // Determinant should be +1.
            let det = r[0] * (r[4] * r[8] - r[5] * r[7])
                - r[1] * (r[3] * r[8] - r[5] * r[6])
                + r[2] * (r[3] * r[7] - r[4] * r[6]);
            assert!(
                (det - 1.0).abs() < 1e-9,
                "det(R) = {det} for omega = {omega:?}"
            );
        }
    }

    #[test]
    fn interpolation_endpoints_so3() {
        let r0 = exp_so3(&[0.1, 0.2, 0.3]);
        let r1 = exp_so3(&[0.5, -0.3, 0.8]);

        let at_0 = geodesic_interpolation_so3(&r0, &r1, 0.0);
        let at_1 = geodesic_interpolation_so3(&r0, &r1, 1.0);

        assert!(
            mat_dist(&at_0, &r0) < 1e-9,
            "interp(t=0) != r0, dist = {}",
            mat_dist(&at_0, &r0)
        );
        assert!(
            mat_dist(&at_1, &r1) < 1e-9,
            "interp(t=1) != r1, dist = {}",
            mat_dist(&at_1, &r1)
        );
    }

    #[test]
    fn interpolation_midpoint_so3() {
        let r0 = IDENTITY_SO3;
        let r1 = exp_so3(&[0.0, 0.0, PI / 2.0]); // 90 degrees about z

        let mid = geodesic_interpolation_so3(&r0, &r1, 0.5);
        let expected = exp_so3(&[0.0, 0.0, PI / 4.0]); // 45 degrees about z

        assert!(
            mat_dist(&mid, &expected) < 1e-9,
            "midpoint should be 45-degree rotation, dist = {}",
            mat_dist(&mid, &expected)
        );
    }

    // -- SE(3) tests -------------------------------------------------------

    #[test]
    fn exp_log_roundtrip_se3() {
        let cases: &[Twist] = &[
            [0.1, 0.2, 0.3, 1.0, -0.5, 0.2],
            [0.0, 0.0, 0.0, 1.0, 2.0, 3.0], // pure translation
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  // pure rotation
            [0.5, -0.3, 0.8, 0.1, 0.2, 0.3],
        ];

        for xi in cases {
            let p = exp_se3(xi);
            let recovered = log_se3(&p);
            let p2 = exp_se3(&recovered);

            assert!(
                mat_dist(&p.0, &p2.0) < 1e-9,
                "rotation roundtrip failed for xi = {xi:?}, dist = {}",
                mat_dist(&p.0, &p2.0)
            );
            let t_dist: f64 = p
                .1
                .iter()
                .zip(p2.1.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();
            assert!(
                t_dist < 1e-9,
                "translation roundtrip failed for xi = {xi:?}, dist = {t_dist}"
            );
        }
    }

    #[test]
    fn pure_translation_se3() {
        let xi: Twist = [0.0, 0.0, 0.0, 3.0, -1.0, 2.0];
        let (rot, trans) = exp_se3(&xi);

        // Rotation should be identity.
        assert!(
            mat_dist(&rot, &IDENTITY_SO3) < TOL,
            "pure translation should have identity rotation"
        );

        // Translation should match linear velocity.
        for i in 0..3 {
            assert!(
                (trans[i] - xi[i + 3]).abs() < TOL,
                "translation[{i}] = {}, expected {}",
                trans[i],
                xi[i + 3]
            );
        }
    }

    #[test]
    fn identity_se3_roundtrip() {
        let xi = log_se3(&IDENTITY_SE3);
        let norm: f64 = xi.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(norm < TOL, "log(identity) should be zero, got {xi:?}");
    }

    #[test]
    fn interpolation_endpoints_se3() {
        let p0 = exp_se3(&[0.1, 0.2, 0.3, 1.0, -0.5, 0.2]);
        let p1 = exp_se3(&[0.5, -0.3, 0.8, 0.1, 0.2, 0.3]);

        let at_0 = geodesic_interpolation_se3(&p0, &p1, 0.0);
        let at_1 = geodesic_interpolation_se3(&p0, &p1, 1.0);

        assert!(
            mat_dist(&at_0.0, &p0.0) < 1e-9,
            "SE3 interp(t=0) rotation != p0"
        );
        let t0_dist: f64 = at_0
            .1
            .iter()
            .zip(p0.1.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        assert!(t0_dist < 1e-9, "SE3 interp(t=0) translation != p0");

        assert!(
            mat_dist(&at_1.0, &p1.0) < 1e-9,
            "SE3 interp(t=1) rotation != p1"
        );
        let t1_dist: f64 = at_1
            .1
            .iter()
            .zip(p1.1.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        assert!(t1_dist < 1e-9, "SE3 interp(t=1) translation != p1");
    }

    #[test]
    fn compose_inverse_is_identity() {
        let p = exp_se3(&[0.3, -0.5, 0.1, 2.0, -1.0, 0.5]);
        let p_inv = inverse_se3(&p);
        let result = compose_se3(&p, &p_inv);

        assert!(
            mat_dist(&result.0, &IDENTITY_SO3) < 1e-9,
            "p * p^-1 rotation != I"
        );
        let t_norm: f64 = result.1.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(t_norm < 1e-9, "p * p^-1 translation != 0, got {t_norm}");
    }

    #[test]
    fn orthogonality_preserved_through_interpolation() {
        let r0 = exp_so3(&[0.3, 0.5, -0.2]);
        let r1 = exp_so3(&[1.0, -0.5, 0.3]);

        for &t in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            let r = geodesic_interpolation_so3(&r0, &r1, t);
            let rt = transpose3(&r);
            let product = mat_mul3(&r, &rt);
            assert!(
                mat_dist(&product, &IDENTITY_SO3) < 1e-9,
                "R*R^T != I at t = {t}, dist = {}",
                mat_dist(&product, &IDENTITY_SO3)
            );
        }
    }

    #[test]
    fn near_pi_rotation_roundtrip() {
        // Rotation very close to pi radians -- tests the near-pi branch of log_so3.
        let omega: AxisAngle = [PI * 0.99, 0.0, 0.0];
        let r = exp_so3(&omega);
        let recovered = log_so3(&r);
        let r2 = exp_so3(&recovered);
        assert!(
            mat_dist(&r, &r2) < 1e-8,
            "near-pi roundtrip failed, dist = {}",
            mat_dist(&r, &r2)
        );
    }
}
