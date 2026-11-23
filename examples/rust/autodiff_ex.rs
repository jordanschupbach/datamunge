// CMD: just TARGET=autodiff_ex run-rust
// Automatic differentiation: forward mode (Dual), reverse mode (Tape/Var), Hessians (HyperDual).
//
// Note: the C++ autodiff_ex.cpp additionally demonstrates the generic template drivers
// (derivative/gradient_forward/gradient_reverse/jacobian_forward/hessian), which can't be
// bound. This port demonstrates the same ideas with the bound named-method API.
// Var::index() is the Var's position in the adjoint vector Tape::backward returns.
use datamunge::*;
use datamunge_rust_examples::N;

fn rosenbrock_dual(x0: &Dual, x1: &Dual) -> Result<Dual> {
    let a = Dual::new_with_derivative(1.0, 0.0)?.subtract(x0)?;
    let b = x1.subtract(&x0.multiply(x0)?)?;
    a.multiply(&a)?.add(&b.multiply(&b)?.multiply_scalar(100.0)?)
}

fn rosenbrock_value(x0: f64, x1: f64) -> f64 {
    (1.0 - x0).powi(2) + 100.0 * (x1 - x0 * x0).powi(2)
}

/// Builds the Rosenbrock graph on `tape`; returns the output Var.
fn rosenbrock_var(tape: &Tape, x0: &Var, x1: &Var) -> Result<Var> {
    let a = Var::new(tape, 1.0)?.subtract(x0)?;
    let b = x1.subtract(&x0.multiply(x0)?)?;
    a.multiply(&a)?.add(&b.multiply(&b)?.multiply_scalar(100.0)?)
}

fn rosenbrock_hyper_dual(x0: &HyperDual, x1: &HyperDual) -> Result<HyperDual> {
    let a = HyperDual::new_with_eps1eps2(1.0, 0.0, 0.0, 0.0)?.subtract(x0)?;
    let b = x1.subtract(&x0.multiply(x0)?)?;
    a.multiply(&a)?.add(&b.multiply(&b)?.multiply_scalar(100.0)?)
}

fn adjoint(adjoints: &[f64], v: &Var) -> Result<f64> {
    Ok(adjoints[v.index()?])
}

fn main() -> datamunge::Result<()> {
    println!("=================== Forward mode: scalar derivative ===================");
    {
        let x0 = Dual::new_with_derivative(2.0, 1.0)?; // seed derivative = 1 to read df/dx directly
        let f = x0.multiply(&x0)?.multiply(&x0)?.subtract(&x0.multiply_scalar(2.0)?)?; // x^3 - 2x
        println!("f(x) = x^3 - 2x, f'(2) = {} (exact: 10)", N(f.derivative()?));
    }

    println!("\n=================== Reverse mode: build a graph by hand ===================");
    {
        let tape = Tape::new()?;
        let a = Var::new(&tape, 2.0)?;
        let b = Var::new(&tape, 3.0)?;
        let y = a.multiply(&b)?.add(&a.sin()?)?;
        println!("y = a*b + sin(a) at a=2, b=3 -> y = {}", N(y.value()?));
        let adjoints = tape.backward(&y)?;
        println!("dy/da = {} (exact: b + cos(a))", N(adjoint(&adjoints, &a)?));
        println!("dy/db = {} (exact: a)", N(adjoint(&adjoints, &b)?));
    }

    println!("\n=================== Forward vs reverse mode agree on the Rosenbrock function ===================");
    {
        let (px, py) = (0.0, 0.0);
        // Forward mode: one Dual pass per partial derivative, seeding the direction of interest.
        let gx = rosenbrock_dual(&Dual::new_with_derivative(px, 1.0)?, &Dual::new_with_derivative(py, 0.0)?)?.derivative()?;
        let gy = rosenbrock_dual(&Dual::new_with_derivative(px, 0.0)?, &Dual::new_with_derivative(py, 1.0)?)?.derivative()?;
        // Reverse mode: one Tape/Var pass computes every partial at once.
        let tape = Tape::new()?;
        let vx = Var::new(&tape, px)?;
        let vy = Var::new(&tape, py)?;
        let grad = tape.backward(&rosenbrock_var(&tape, &vx, &vy)?)?;
        println!("f(0,0) = {}", N(rosenbrock_value(px, py)));
        println!("gradient (forward mode): [{}, {}]", N(gx), N(gy));
        println!("gradient (reverse mode): [{}, {}]", N(adjoint(&grad, &vx)?), N(adjoint(&grad, &vy)?));
    }

    println!("\n=================== Jacobian of a vector-valued function ===================");
    // f(x,y) = [x^2, xy, y^3] at (2,3) -- one Dual pass per (output, input) pair.
    {
        let vector_fn = |x: &Dual, y: &Dual| -> Result<[Dual; 3]> {
            Ok([x.multiply(x)?, x.multiply(y)?, y.multiply(y)?.multiply(y)?])
        };
        let mut jac = [[0.0; 2]; 3];
        for (col, sx, sy) in [(0, 1.0, 0.0), (1, 0.0, 1.0)] {
            let outputs = vector_fn(&Dual::new_with_derivative(2.0, sx)?, &Dual::new_with_derivative(3.0, sy)?)?;
            for (row, output) in outputs.iter().enumerate() {
                jac[row][col] = output.derivative()?;
            }
        }
        println!("f(x,y) = [x^2, xy, y^3] at (2,3), Jacobian:");
        for row in &jac {
            println!("  [{}, {}]", N(row[0]), N(row[1]));
        }
    }

    println!("\n=================== Hessian via second-order forward mode (HyperDual) ===================");
    // One HyperDual pass per (i, j) pair reads off d^2f/dxi dxj directly from eps1eps2.
    {
        let (mx, my) = (1.0, 1.0); // the Rosenbrock function's minimum
        let seeds = [(1.0, 0.0), (0.0, 1.0)];
        let mut h = [[0.0; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                let (e1, e2) = (seeds[i], seeds[j]);
                h[i][j] = rosenbrock_hyper_dual(
                    &HyperDual::new_with_eps1eps2(mx, e1.0, e2.0, 0.0)?,
                    &HyperDual::new_with_eps1eps2(my, e1.1, e2.1, 0.0)?,
                )?
                .eps1eps2()?;
            }
        }
        println!("Hessian of the Rosenbrock function at its minimum (1,1):");
        for row in &h {
            println!("  [{}, {}]", N(row[0]), N(row[1]));
        }
    }

    println!("\n=================== Gradient descent driven by reverse-mode gradients ===================");
    {
        let mut point = [-1.2, 1.0]; // the classic Rosenbrock start
        const LEARNING_RATE: f64 = 0.001;
        const N_STEPS: usize = 2000;
        for step in 0..N_STEPS {
            let tape = Tape::new()?;
            let vx = Var::new(&tape, point[0])?;
            let vy = Var::new(&tape, point[1])?;
            let vf = rosenbrock_var(&tape, &vx, &vy)?;
            let grad = tape.backward(&vf)?;
            let loss = vf.value()?;
            point[0] -= LEARNING_RATE * adjoint(&grad, &vx)?;
            point[1] -= LEARNING_RATE * adjoint(&grad, &vy)?;
            if step == 0 || step == N_STEPS - 1 {
                println!("step {step}: loss = {}, x = [{}, {}]", N(loss), N(point[0]), N(point[1]));
            }
        }
        println!("(true minimum is at [1, 1] with loss 0)");
    }
    Ok(())
}
