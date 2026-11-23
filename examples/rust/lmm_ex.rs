// CMD: just TARGET=lmm_ex run-rust
// Linear mixed models: random intercept (penguins) and random intercept + slope (simulated schools).
use datamunge::*;
use datamunge_rust_examples::{DotNetRandom, N};
use std::f64::consts::PI;

/// (0, 1]: avoids log(0) in the Box-Muller transform.
fn uniform(rng: &mut DotNetRandom) -> f64 {
    1.0 - rng.next_double()
}

fn gauss(rng: &mut DotNetRandom) -> f64 {
    (-2.0 * uniform(rng).ln()).sqrt() * (2.0 * PI * uniform(rng)).cos()
}

fn main() -> datamunge::Result<()> {
    // .NET's System.Random, so the simulated data matches the C# example's.
    let mut rng = DotNetRandom::new(2024);

    println!("=================== Random intercept on a real dataset (penguins) ===================");
    LMM::new(&data::penguins()?.expect("penguins"), "body_mass_g ~ flipper_length_mm + bill_length_mm + (1 | species)")?
        .print_summary()?;

    println!("\n=================== Random intercept + slope on a simulated multi-school dataset ===================");
    // 30 schools, ~25 students each: score depends on study_hours, but both the baseline score
    // and the return on an extra hour of study vary by school -- the textbook case for a
    // random-intercept-and-slope model.
    const N_SCHOOLS: usize = 30;
    let school_intercept: Vec<f64> = (0..N_SCHOOLS).map(|_| gauss(&mut rng) * 6.0).collect();
    let school_slope: Vec<f64> = (0..N_SCHOOLS).map(|_| gauss(&mut rng) * 1.2).collect();
    const TRUE_INTERCEPT: f64 = 60.0;
    const TRUE_SLOPE: f64 = 3.0;

    let (mut school, mut study_hours, mut score) = (Vec::new(), Vec::new(), Vec::new());
    for s in 0..N_SCHOOLS {
        let n_students = 15 + rng.next_range(0, 21);
        for _ in 0..n_students {
            let hours = rng.next_double() * 10.0;
            school.push(s as f64);
            study_hours.push(hours);
            score.push(TRUE_INTERCEPT + school_intercept[s] + (TRUE_SLOPE + school_slope[s]) * hours + gauss(&mut rng) * 4.0);
        }
    }
    let df = DataFrame::new()?;
    df.add_numeric_column("school", &school)?;
    df.add_numeric_column("study_hours", &study_hours)?;
    df.add_numeric_column("score", &score)?;

    let model = LMM::new(&df, "score ~ study_hours + (1 + study_hours | school)")?;
    model.print_summary()?;

    println!(
        "\nTrue generating values: intercept={}, slope={}, random-intercept SD=6.0, random-slope SD=1.2, residual SD=4.0",
        N(TRUE_INTERCEPT),
        N(TRUE_SLOPE)
    );

    println!("\n--- Best Linear Unbiased Predictors (BLUPs) for a few schools ---");
    let labels = model.group_labels()?;
    for (idx, label) in labels.iter().enumerate().take(3) {
        let re = model.random_effects_for_group(idx)?;
        println!("school {label}: intercept shift={}, slope shift={}", N(re[0]), N(re[1]));
    }

    println!("\n--- Prediction: population-level vs. school-adjusted ---");
    let population = DataFrame::new()?;
    let school0 = DataFrame::new()?;
    population.add_numeric_column("study_hours", &[5.0])?;
    school0.add_numeric_column("study_hours", &[5.0])?;
    school0.add_numeric_column("school", &[0.0])?;
    println!("5 study hours, unseen school:      {} (fixed effects only)", N(model.predict(&population)?[0]));
    println!("5 study hours, school 0 (known):    {} (fixed effects + school 0's BLUP)", N(model.predict(&school0)?[0]));
    Ok(())
}
