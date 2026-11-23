// CMD: just TARGET=glmm_ex run-rust
// Generalized linear mixed models: a binomial GLMM on penguins and a Poisson GLMM on simulated store data.
use datamunge::*;
use datamunge_rust_examples::{DotNetRandom, N};
use std::f64::consts::PI;

fn main() -> datamunge::Result<()> {
    // .NET's System.Random, so the simulated data matches the C# example's.
    let mut rng = DotNetRandom::new(4242);
    let gauss = |rng: &mut DotNetRandom| {
        let (u1, u2) = (1.0 - rng.next_double(), rng.next_double()); // 1 - u keeps log() away from 0
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    };

    println!("=================== Binomial (logistic) mixed model on a real dataset (penguins) ===================");
    // Predicting sex from body mass, grouped by island: unlike species (which is almost perfectly
    // confounded with island in this dataset -- Gentoo penguins occur on Biscoe island only), sex
    // is not tied to location, so this is a well-behaved fit.
    {
        let penguins = data::penguins()?.expect("penguins");
        let (mut is_male, mut body_mass, mut island) = (Vec::new(), Vec::new(), Vec::new());
        'rows: for i in 0..penguins.nrows()? {
            for c in ["sex", "body_mass_g", "island"] {
                if penguins.is_null(c, i)? {
                    continue 'rows;
                }
            }
            is_male.push(if penguins.string_at("sex", i)? == "male" { 1.0 } else { 0.0 });
            body_mass.push(penguins.numeric_at("body_mass_g", i)?);
            island.push(penguins.string_at("island", i)?);
        }
        let sex_df = DataFrame::new()?;
        sex_df.add_numeric_column("is_male", &is_male)?;
        sex_df.add_numeric_column("body_mass_g", &body_mass)?;
        sex_df.add_string_column("island", &island)?;
        GLMM::new_with_family(&sex_df, "is_male ~ body_mass_g + (1 | island)", "binomial")?.print_summary()?;
    }

    println!("\n=================== Poisson mixed model on simulated multi-site count data ===================");
    // 25 stores, ~20 days each: daily visit counts depend on a promo intensity score, but both
    // the baseline traffic and the promo's effectiveness vary by store.
    const N_STORES: usize = 25;
    let store_effect: Vec<f64> = (0..N_STORES).map(|_| gauss(&mut rng) * 0.4).collect();
    const TRUE_INTERCEPT: f64 = 2.0;
    const TRUE_SLOPE: f64 = 0.3;

    let (mut store, mut promo, mut visits) = (Vec::new(), Vec::new(), Vec::new());
    for (s, effect) in store_effect.iter().enumerate() {
        let n_days = 15 + rng.next_range(0, 11);
        for _ in 0..n_days {
            let promo_intensity = rng.next_double() * 3.0;
            let lam = (TRUE_INTERCEPT + effect + TRUE_SLOPE * promo_intensity).exp();
            // Knuth's Poisson sampler.
            let (threshold, mut k, mut p) = ((-lam).exp(), 0, 1.0);
            loop {
                k += 1;
                p *= rng.next_double();
                if p <= threshold {
                    break;
                }
            }
            store.push(s as f64);
            promo.push(promo_intensity);
            visits.push((k - 1) as f64);
        }
    }
    let df = DataFrame::new()?;
    df.add_numeric_column("store", &store)?;
    df.add_numeric_column("promo", &promo)?;
    df.add_numeric_column("visits", &visits)?;
    let store_model = GLMM::new_with_family(&df, "visits ~ promo + (1 | store)", "poisson")?;

    store_model.print_summary()?;

    println!(
        "\nTrue generating values: intercept={}, slope={}, random-intercept SD (log scale)=0.4",
        N(TRUE_INTERCEPT),
        N(TRUE_SLOPE)
    );

    println!("\n--- BLUPs for a few stores ---");
    let labels = store_model.group_labels()?;
    for (idx, label) in labels.iter().enumerate().take(3) {
        println!("store {label}: intercept shift={}", N(store_model.random_effects_for_group(idx)?[0]));
    }

    println!("\n--- Prediction: population-level vs. store-adjusted ---");
    let population = DataFrame::new()?;
    let store0 = DataFrame::new()?;
    population.add_numeric_column("promo", &[1.5])?;
    store0.add_numeric_column("promo", &[1.5])?;
    store0.add_numeric_column("store", &[0.0])?;
    println!(
        "promo=1.5, unseen store:   {} expected visits (fixed effects only)",
        N(store_model.predict(&population)?[0])
    );
    println!(
        "promo=1.5, store 0 (known): {} expected visits (fixed effects + store 0's BLUP)",
        N(store_model.predict(&store0)?[0])
    );
    Ok(())
}
