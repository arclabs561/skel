//! Geometric-algebra rotor rotation in 2D.
//!
//! This is an example-local `geomalg` proof. It implements the small
//! Clifford algebra Cl(2, 0), rotates a vector by the sandwich product
//! `R v reverse(R)`, and compares the result with the ordinary 2D rotation
//! matrix.
//!
//! Run: cargo run --example rotor_rotation

#[derive(Clone, Copy, Debug)]
struct Multivector {
    scalar: f64,
    e1: f64,
    e2: f64,
    e12: f64,
}

impl Multivector {
    fn vector(x: f64, y: f64) -> Self {
        Self {
            scalar: 0.0,
            e1: x,
            e2: y,
            e12: 0.0,
        }
    }

    fn rotor(theta: f64) -> Self {
        let half = theta / 2.0;
        Self {
            scalar: half.cos(),
            e1: 0.0,
            e2: 0.0,
            e12: -half.sin(),
        }
    }

    fn reverse(self) -> Self {
        Self {
            e12: -self.e12,
            ..self
        }
    }

    fn xy(self) -> [f64; 2] {
        [self.e1, self.e2]
    }

    fn vector_norm(self) -> f64 {
        (self.e1.powi(2) + self.e2.powi(2)).sqrt()
    }
}

impl std::ops::Mul for Multivector {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let a = self.scalar;
        let b = self.e1;
        let c = self.e2;
        let d = self.e12;
        let e = rhs.scalar;
        let f = rhs.e1;
        let g = rhs.e2;
        let h = rhs.e12;

        Self {
            scalar: a * e + b * f + c * g - d * h,
            e1: a * f + b * e - c * h + d * g,
            e2: a * g + b * h + c * e - d * f,
            e12: a * h + b * g - c * f + d * e,
        }
    }
}

fn main() {
    let theta = std::f64::consts::FRAC_PI_3;
    let v = Multivector::vector(1.0, 0.0);
    let rotor = Multivector::rotor(theta);
    let rotated = rotor * v * rotor.reverse();
    let matrix = matrix_rotate([1.0, 0.0], theta);
    let composed = Multivector::rotor(std::f64::consts::FRAC_PI_6)
        * Multivector::rotor(std::f64::consts::FRAC_PI_4);
    let composed_xy = (composed * v * composed.reverse()).xy();
    let matrix_composed = matrix_rotate([1.0, 0.0], 5.0 * std::f64::consts::PI / 12.0);

    println!("2D Clifford rotor rotation");
    println!("angle: pi/3");
    println!("matrix rotation: [{:.6}, {:.6}]", matrix[0], matrix[1]);
    println!("rotor sandwich:  [{:.6}, {:.6}]", rotated.e1, rotated.e2);
    println!(
        "norm before/after: {:.6} -> {:.6}",
        v.vector_norm(),
        rotated.vector_norm()
    );
    println!(
        "composed rotor pi/6 + pi/4: [{:.6}, {:.6}]",
        composed_xy[0], composed_xy[1]
    );

    assert_close(rotated.xy(), matrix, 1e-12);
    assert_close(composed_xy, matrix_composed, 1e-12);
    assert!((v.vector_norm() - rotated.vector_norm()).abs() < 1e-12);
}

fn matrix_rotate(v: [f64; 2], theta: f64) -> [f64; 2] {
    let (sin, cos) = theta.sin_cos();
    [cos * v[0] - sin * v[1], sin * v[0] + cos * v[1]]
}

fn assert_close(actual: [f64; 2], expected: [f64; 2], tolerance: f64) {
    let err = ((actual[0] - expected[0]).powi(2) + (actual[1] - expected[1]).powi(2)).sqrt();
    assert!(
        err < tolerance,
        "actual={actual:?}, expected={expected:?}, err={err}"
    );
}
