// CMD: just TARGET=timeseries_ex run-rust
// Time series: ARIMA, seasonal ARIMA and exponential smoothing (SES, Holt, Holt-Winters).
use datamunge::*;
use datamunge_rust_examples::{join, DotNetRandom, N};
use std::f64::consts::PI;

/// (0, 1]: avoids log(0) in the Box-Muller transform.
fn uniform(rng: &mut DotNetRandom) -> f64 {
    1.0 - rng.next_double()
}

fn gauss(rng: &mut DotNetRandom) -> f64 {
    (-2.0 * uniform(rng).ln()).sqrt() * (2.0 * PI * uniform(rng)).cos()
}

fn joined(v: &[f64]) -> String {
    join(v, ", ")
}

fn main() -> datamunge::Result<()> {
    println!("=================== ARIMA(1,1,1) on a simulated random walk with drift ===================");
    // .NET's System.Random, so the simulated data matches the C# example's.
    let mut rng = DotNetRandom::new(42);
    let mut y = Vec::new();
    {
        let (mut level, mut prev_shock) = (100.0, 0.0);
        for _ in 0..150 {
            let shock = gauss(&mut rng);
            level += 0.3 + 0.4 * prev_shock + shock; // integrated ARMA(0,1) increments with drift
            y.push(level);
            prev_shock = shock;
        }
    }

    let options = ARIMAOptions::new()?;
    options.set_p(1)?;
    options.set_d(1)?;
    options.set_q(1)?;
    options.set_de_population_size(80)?;
    options.set_de_max_generations(400)?;
    let model = ARIMA::new_with_options(&y, &options)?;

    println!("AR coefficient: {}", N(model.ar_coefficients()?[0]));
    println!("MA coefficient: {}", N(model.ma_coefficients()?[0]));
    println!("sigma^2: {}, AIC: {}, BIC: {}", N(model.sigma2()?), N(model.aic()?), N(model.bic()?));

    let (forecast_values, std_errors) = model.forecast_with_intervals(6)?;
    println!("6-step forecast: {}", joined(&forecast_values));
    println!("forecast std. errors: {}", joined(&std_errors));

    println!("\n=================== SARIMA(1,0,0)(1,1,0)_12 on a seasonal series ===================");
    let mut rng = DotNetRandom::new(7);
    let mut s = Vec::new();
    {
        let mut prev = 0.0;
        for i in 0..120 {
            prev = 0.5 * prev + gauss(&mut rng);
            s.push(20.0 + 0.2 * i as f64 + 5.0 * (2.0 * PI * i as f64 / 12.0).sin() + prev);
        }
    }

    let options2 = ARIMAOptions::new()?;
    options2.set_p(1)?;
    options2.set_seasonal_p(1)?;
    options2.set_seasonal_d(1)?;
    options2.set_seasonal_period(12)?;
    options2.set_de_population_size(100)?;
    options2.set_de_max_generations(500)?;
    let model2 = ARIMA::new_with_options(&s, &options2)?;
    println!(
        "AR coefficient: {}, seasonal AR coefficient: {}",
        N(model2.ar_coefficients()?[0]),
        N(model2.seasonal_ar_coefficients()?[0])
    );
    println!("12-step forecast: {}", joined(&model2.forecast(12)?));

    println!("\n=================== Exponential smoothing: Holt-Winters on retail-style seasonal data ===================");
    let seasonal_shape = [0.8, 0.75, 0.9, 0.95, 1.0, 1.05, 1.1, 1.05, 1.0, 1.1, 1.3, 1.6];
    let mut rng = DotNetRandom::new(11);
    let y2: Vec<f64> =
        (0..48).map(|i| (100.0 + 2.0 * i as f64) * seasonal_shape[i % 12] + gauss(&mut rng) * 3.0).collect();

    let es_options = ExponentialSmoothingOptions::new()?;
    es_options.set_trend(TrendType::ADDITIVE)?;
    es_options.set_seasonal(SeasonalType::MULTIPLICATIVE)?;
    es_options.set_seasonal_period(12)?;
    es_options.set_de_population_size(60)?;
    es_options.set_de_max_generations(300)?;
    let es_model = ExponentialSmoothing::new_with_options(&y2, &es_options)?;
    println!("alpha={} beta={} gamma={}", N(es_model.alpha()?), N(es_model.beta()?), N(es_model.gamma()?));
    println!("sigma^2: {}, AIC: {}", N(es_model.sigma2()?), N(es_model.aic()?));
    println!("12-month forecast: {}", joined(&es_model.forecast(12)?));

    println!("\n=================== Simple exponential smoothing vs. Holt's linear trend ===================");
    let flat = [50.2, 49.8, 50.5, 49.6, 50.1, 50.3, 49.9, 50.0, 50.4, 49.7];

    let ses = ExponentialSmoothing::new_with_options(&flat, &ExponentialSmoothingOptions::new()?)?;
    println!("SES alpha= {}", N(ses.alpha()?));
    println!("SES 5-step forecast: {}", joined(&ses.forecast(5)?));

    let holt_options = ExponentialSmoothingOptions::new()?;
    holt_options.set_trend(TrendType::ADDITIVE)?;
    let holt = ExponentialSmoothing::new_with_options(&flat, &holt_options)?;
    println!("Holt alpha={} beta={}", N(holt.alpha()?), N(holt.beta()?));
    println!("Holt 5-step forecast: {}", joined(&holt.forecast(5)?));
    Ok(())
}
