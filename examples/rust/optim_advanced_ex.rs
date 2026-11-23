// CMD: just TARGET=optim_advanced_ex run-rust
// Advanced datamunge optimizers (ProximalGradient, FISTA, LevenbergMarquardt, Newton, TrustRegionNewton,
// AugmentedLagrangian, SQP, InteriorPoint, BayesianOptimization, RBFGaussianProcessSurrogate) from Rust.
//
// Every one of these optimizers takes an interface (ProximalFunction / HessianFunction /
// EqualityConstrainedFunction / InequalityConstrainedFunction / ResidualFunction, or
// ArbitraryFunction for BayesianOptimization) that the C++ library leaves abstract: callers
// implement the objective. From Rust, implement the interface's `XImpl` trait and wrap it with
// `X::from_impl`; the optimizer then calls the Rust methods. Coordinates are updated in place
// through a `&mut Vec<f64>` (the C++ parameter is a non-const std::vector<double>&).
use datamunge::*;
use datamunge_rust_examples::{net, net_bool, net_g};

/// f(x, y) = (x - 1)^2 + 10 (y + 2)^2 with its exact gradient and Hessian.
struct Quadratic;

impl HessianFunctionImpl for Quadratic {
    fn evaluate(&self, _this: &HessianFunction, c: Vec<f64>) -> Result<f64> {
        Ok((c[0] - 1.0).powi(2) + 10.0 * (c[1] + 2.0).powi(2))
    }

    fn gradient(&self, _this: &HessianFunction, c: Vec<f64>) -> Result<Vec<f64>> {
        Ok(vec![2.0 * (c[0] - 1.0), 20.0 * (c[1] + 2.0)])
    }

    fn hessian(&self, _this: &HessianFunction, _c: Vec<f64>) -> Result<Vec<Vec<f64>>> {
        Ok(vec![vec![2.0, 0.0], vec![0.0, 20.0]])
    }
}

/// Residuals of y = a * exp(b t) against exact data from a = 2, b = 0.5.
struct Exponential {
    ts: Vec<f64>,
}

impl ResidualFunctionImpl for Exponential {
    fn residuals(&self, _this: &ResidualFunction, p: Vec<f64>) -> Result<Vec<f64>> {
        Ok(self.ts.iter().map(|t| p[0] * (p[1] * t).exp() - 2.0 * (0.5 * t).exp()).collect())
    }

    fn jacobian(&self, _this: &ResidualFunction, p: Vec<f64>) -> Result<Vec<Vec<f64>>> {
        Ok(self.ts.iter().map(|t| vec![(p[1] * t).exp(), p[0] * t * (p[1] * t).exp()]).collect())
    }
}

/// Leaves out gradient() and hessian(): the C++ pure virtuals fail when the optimizer calls them.
struct Incomplete;

impl HessianFunctionImpl for Incomplete {
    fn evaluate(&self, _this: &HessianFunction, _c: Vec<f64>) -> Result<f64> {
        Ok(0.0)
    }
}

fn main() -> datamunge::Result<()> {
    println!("=================== RBFGaussianProcessSurrogate (concrete BayesianSurrogate) ===================");
    // Fit a 1D surrogate to a simple bowl y = (x-1)^2 sampled at a few points,
    // then check the acquisition function favors an unexplored region.
    {
        let xs = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0];
        let ys: Vec<f64> = xs.iter().map(|x| (x - 1.0) * (x - 1.0)).collect();
        let surrogate = RBFGaussianProcessSurrogate::new_with_noise(1.0, 1e-6)?;
        let points: Vec<Vec<f64>> = xs.iter().map(|&x| vec![x]).collect();
        surrogate.fit(&points, &ys)?;
        let incumbent = ys.iter().copied().fold(0.0, f64::min); // best observed value so far
        println!("incumbent (best observed y): {}", net(incumbent));
        for x in [0.25, 1.0, 2.5] {
            println!("acquisition(x={x:.2}) = {:.6}", surrogate.acquisition(&[x], incumbent)?);
        }
    }

    println!("\n=================== Options structs + optimizer construction (all 10 new classes) ===================");
    // Every new Options struct and optimizer class constructs cleanly, proving
    // the binding for the whole batch is wired correctly.
    let pg_options = ProximalGradientOptions::new()?;
    pg_options.set_step_size(0.05)?;
    let pg = ProximalGradient::new_with_options(&pg_options)?;
    println!("ProximalGradient: {pg:?}");
    let fista = FISTA::new_with_options(&FISTAOptions::new()?)?;
    println!("FISTA: {fista:?}");
    let lm_options = LevenbergMarquardtOptions::new()?;
    lm_options.set_max_iterations(200)?;
    let lm = LevenbergMarquardt::new_with_options(&lm_options)?;
    println!("LevenbergMarquardt: {lm:?}");
    let newton = Newton::new_with_options(&NewtonOptions::new()?)?;
    println!("Newton: {newton:?}");
    let trn = TrustRegionNewton::new_with_options(&TrustRegionNewtonOptions::new()?)?;
    println!("TrustRegionNewton: {trn:?}");
    let al = AugmentedLagrangian::new_with_options(&AugmentedLagrangianOptions::new()?)?;
    println!("AugmentedLagrangian: {al:?}");
    let sqp = SQP::new_with_options(&SQPOptions::new()?)?;
    println!("SQP: {sqp:?}");
    let ip = InteriorPoint::new_with_options(&InteriorPointOptions::new()?)?;
    println!("InteriorPoint: {ip:?}");
    let bo_options = BayesianOptimizationOptions::new()?;
    bo_options.set_initial_samples(8)?;
    let bo = BayesianOptimization::new_with_options(&bo_options)?;
    println!("BayesianOptimization: {bo:?}");

    println!("\n=================== Rust objectives (implementing the C++ interfaces) ===================");
    // Newton's method on f(x, y) = (x - 1)^2 + 10 (y + 2)^2 with exact gradient and Hessian.
    {
        let mut start = vec![5.0, 5.0];
        let value = newton.optimize(&HessianFunction::from_impl(Quadratic)?, &mut start)?;
        println!(
            "Newton: f = {} at ({:.6}, {:.6}) (exact minimum 0 at (1, -2))",
            net_g(value, 6),
            start[0],
            start[1]
        );
    }

    // Levenberg-Marquardt fitting y = a * exp(b t) to exact data from a = 2, b = 0.5.
    {
        let mut parameters = vec![1.0, 0.1];
        let residuals = ResidualFunction::from_impl(Exponential { ts: vec![0.0, 0.5, 1.0, 1.5, 2.0] })?;
        let sse = lm.optimize(&residuals, &mut parameters)?;
        println!(
            "LevenbergMarquardt: residual = {}, a = {:.6}, b = {:.6} (exact: 2, 0.5)",
            net_g(sse, 3),
            parameters[0],
            parameters[1]
        );
    }

    // An interface method an implementation leaves out fails when the optimizer calls it.
    {
        let mut origin = vec![0.0, 0.0];
        let result = newton.optimize(&HessianFunction::from_impl(Incomplete)?, &mut origin);
        let mentions = result.as_ref().is_err_and(|e| e.to_string().contains("gradient"));
        println!("missing gradient(): ok={}, error mentions it: {}", net_bool(result.is_ok()), net_bool(mentions));
    }
    Ok(())
}
