// CMD: just TARGET=gaussian_process_regression_iris_ex run-rust
// Gaussian process regression of Petal.Length on Petal.Width (iris), with confidence intervals.
use datamunge::*;

const FORMULA: &str = "Petal.Length ~ Petal.Width";

fn main() -> datamunge::Result<()> {
    let iris = data::iris()?.expect("iris");
    println!("iris: {} rows x {} cols", iris.nrows()?, iris.ncols()?);
    println!("formula: {FORMULA}\n");

    // Both the length scale and the noise ratio are auto-selected by maximizing the exact log
    // marginal likelihood.
    let model = GaussianProcessRegression::new(&iris, FORMULA)?;
    model.print_summary()?;

    model.plot_fit(&iris)?.save("gpr_iris_fit.svg")?;
    model.plot_length_scale_profile()?.save("gpr_iris_length_scale_profile.svg")?;
    println!("\nSaved gpr_iris_fit.svg and gpr_iris_length_scale_profile.svg");

    // Unlike every other regressor in this suite, a GP gives a genuine posterior confidence
    // interval at every point -- including far outside the training data, where it should widen
    // substantially as the model's uncertainty grows.
    let query = DataFrame::new()?;
    query.add_numeric_column("Petal.Width", &[0.2, 1.3, 2.5, 10.0])?;
    let detail = model.predict_frame_with_interval_kind(&query, "confidence")?.expect("predictions");
    println!("\nPredictions with 95% confidence intervals:");
    println!("{detail}");
    println!(
        "(Petal.Width=10.0 is far outside the training range [0.1, 2.5] -- note how much wider its\n \
         interval is than the in-range predictions.)"
    );
    Ok(())
}
