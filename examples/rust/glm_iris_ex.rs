// CMD: just TARGET=glm_iris_ex run-rust
// Generalized linear models on iris: logistic regression (binomial) and Poisson regression.
use datamunge::*;
use datamunge_rust_examples::N;

fn main() -> datamunge::Result<()> {
    let iris = data::iris()?.expect("iris");
    let rows: Vec<usize> = (0..iris.nrows()?).collect();
    let column = |name: &str, rows: &[usize]| rows.iter().map(|&i| iris.numeric_at(name, i)).collect::<Result<Vec<_>>>();

    // Logistic regression: versicolor vs virginica only -- setosa is perfectly separable from
    // the other two on these predictors, which sends logistic regression's coefficients toward
    // +/-infinity (a genuine degeneracy of the method, not a bug) -- so this is the well-behaved
    // binary split for a demo.
    let mut kept = Vec::new();
    let mut is_virginica = Vec::new();
    for &i in &rows {
        let species = iris.string_at("Species", i)?;
        if species == "versicolor" || species == "virginica" {
            kept.push(i);
            is_virginica.push(if species == "virginica" { 1.0 } else { 0.0 });
        }
    }
    let petal_length = column("Petal.Length", &kept)?;
    let petal_width = column("Petal.Width", &kept)?;

    let sub = DataFrame::new()?;
    sub.add_numeric_column("Petal.Length", &petal_length)?;
    sub.add_numeric_column("Petal.Width", &petal_width)?;
    sub.add_numeric_column("is_virginica", &is_virginica)?;

    println!("=================== Logistic regression (binomial, logit link) ===================");
    let logit = GLM::new_with_family(&sub, "is_virginica ~ Petal.Length + Petal.Width", "binomial")?;
    logit.print_summary()?;

    let fitted = logit.fitted_values()?;
    let n = is_virginica.len();
    let correct = fitted.iter().zip(&is_virginica).filter(|(f, y)| (**f >= 0.5) == (**y >= 0.5)).count();
    println!("\nResubstitution accuracy at 0.5 threshold: {}%", N(100.0 * correct as f64 / n as f64));

    logit.save_diagnostic_plots("glm_logistic_iris")?;
    println!("\nSaved glm_logistic_iris_{{residuals_vs_fitted,normal_qq,scale_location,residuals_vs_leverage}}.svg");

    // Predicted-probability curve across Petal.Length, with Petal.Width held at its mean -- the
    // classic sigmoid shape of a fitted logistic regression, with a 95% confidence band.
    {
        const GRID_N: usize = 100;
        let width_mean = petal_width.iter().sum::<f64>() / petal_width.len() as f64;
        let pl_min = petal_length.iter().copied().fold(f64::INFINITY, f64::min) - 0.3;
        let pl_max = petal_length.iter().copied().fold(f64::NEG_INFINITY, f64::max) + 0.3;
        let grid = DataFrame::new()?;
        let lengths: Vec<f64> = (0..GRID_N).map(|i| pl_min + (pl_max - pl_min) * i as f64 / (GRID_N - 1) as f64).collect();
        grid.add_numeric_column("Petal.Length", &lengths)?;
        grid.add_numeric_column("Petal.Width", &vec![width_mean; GRID_N])?;
        let curve_frame = logit.predict_frame_with_interval_kind(&grid, "confidence")?.expect("curve");
        println!("\nPredicted-probability curve (first 5 rows):");
        println!("{}", curve_frame.to_string_with_max_rows(5)?);
    }

    // Poisson regression, for contrast: same IRLS engine, different family/link.
    println!("\n=================== Poisson regression (log link) ===================");
    let count_data = DataFrame::new()?;
    count_data.add_numeric_column("Sepal.Width", &column("Sepal.Width", &rows)?)?;
    count_data.add_numeric_column("Petal.Length", &column("Petal.Length", &rows)?)?;
    let counts: Vec<f64> = column("Sepal.Length", &rows)?.iter().map(|x| (x + 0.5).floor()).collect();
    count_data.add_numeric_column("count", &counts)?;
    GLM::new_with_family(&count_data, "count ~ Sepal.Width + Petal.Length", "poisson")?.print_summary()?;
    Ok(())
}
