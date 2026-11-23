// CMD: just TARGET=kernel_regression_iris_ex run-rust
// Nadaraya-Watson kernel regression on iris: CV-selected bandwidth vs. too-small and too-large ones.
use datamunge::*;
use datamunge_rust_examples::N;

const FORMULA: &str = "Petal.Length ~ Petal.Width";

fn main() -> datamunge::Result<()> {
    let iris = data::iris()?.expect("iris");
    println!("iris: {} rows x {} cols", iris.nrows()?, iris.ncols()?);
    println!("formula: {FORMULA}\n");

    let model = KernelRegression::new(&iris, FORMULA)?; // bandwidth auto-selected via leave-one-out CV
    model.print_summary()?;

    model.plot_fit(&iris)?.save("kernel_regression_iris_fit.svg")?;
    model.plot_cv_curve()?.save("kernel_regression_iris_cv.svg")?;
    println!("\nSaved kernel_regression_iris_fit.svg and kernel_regression_iris_cv.svg");

    // A too-small and a too-large bandwidth, for comparison -- the classic kernel regression
    // bias-variance story. KernelRegression::new_with_<last param>(data, formula, kernel, bandwidth,
    // n_bandwidth, standardize)
    let small = KernelRegression::new_with_bandwidth(&iris, FORMULA, "gaussian", 0.05)?;
    println!(
        "\nbandwidth=0.05 (too small): LOO R-squared={}  LOO RMSE={}",
        N(small.r_squared()?),
        N(small.rmse()?)
    );
    small.plot_fit(&iris)?.save("kernel_regression_iris_fit_small_bandwidth.svg")?;

    let large = KernelRegression::new_with_bandwidth(&iris, FORMULA, "gaussian", 5.0)?;
    println!(
        "bandwidth=5.0 (too large):  LOO R-squared={}  LOO RMSE={}",
        N(large.r_squared()?),
        N(large.rmse()?)
    );
    large.plot_fit(&iris)?.save("kernel_regression_iris_fit_large_bandwidth.svg")?;

    println!(
        "bandwidth={} (CV-selected): LOO R-squared={}  LOO RMSE={}",
        N(model.bandwidth()?),
        N(model.r_squared()?),
        N(model.rmse()?)
    );
    println!("\nSaved kernel_regression_iris_fit_small_bandwidth.svg and kernel_regression_iris_fit_large_bandwidth.svg");
    Ok(())
}
