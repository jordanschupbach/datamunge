// CMD: just TARGET=lm_ex run-rust
// Linear model: mpg ~ hp + wt + transmission, its ANOVA table and predictions.
use datamunge::*;

fn main() -> datamunge::Result<()> {
    let cars = DataFrame::new()?;
    cars.add_numeric_column("hp", &[110.0, 110.0, 93.0, 110.0, 175.0, 105.0, 245.0, 62.0, 95.0, 123.0])?;
    cars.add_numeric_column("wt", &[2.62, 2.875, 2.32, 3.215, 3.44, 3.46, 3.57, 3.19, 3.15, 3.44])?;
    cars.add_string_column(
        "transmission",
        &["manual", "manual", "manual", "automatic", "automatic", "automatic", "automatic", "automatic", "automatic", "automatic"],
    )?;
    cars.add_numeric_column("mpg", &[21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2])?;

    println!("Fitting: mpg ~ hp + wt + transmission\n");
    let model = LM::new(&cars, "mpg ~ hp + wt + transmission")?;
    model.print_summary()?;

    println!("\nSequential ANOVA:\n{}", model.anova()?.expect("anova"));

    let newcars = DataFrame::new()?;
    newcars.add_numeric_column("hp", &[150.0, 90.0])?;
    newcars.add_numeric_column("wt", &[3.0, 2.5])?;
    newcars.add_string_column("transmission", &["manual", "automatic"])?;
    println!(
        "\nPredictions with 95% confidence intervals:\n{}",
        model.predict_frame_with_interval_kind(&newcars, "confidence")?.expect("predictions")
    );

    model.save_diagnostic_plots("lm_ex_diagnostics")?;
    println!("\nSaved diagnostic plots as lm_ex_diagnostics_*.svg");
    Ok(())
}
