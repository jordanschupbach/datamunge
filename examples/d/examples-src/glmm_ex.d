module app;

import std.stdio : writeln;
import std.random : Random, uniform;
import std.math : sqrt, log, cos, exp, PI;
import datamunge;

DVector dv(double[] t) {
  auto v = new DVector();
  foreach (x; t) v.push_back(x);
  return v;
}

SVector sv(string[] t) {
  auto v = new SVector();
  foreach (x; t) v.push_back(x);
  return v;
}

double gauss(ref Random rng) {
  double u1 = uniform(0.0, 1.0, rng);
  double u2 = uniform(0.0, 1.0, rng);
  return sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
}

void main() {
  writeln("=================== Binomial (logistic) mixed model on a real dataset (penguins) ===================");
  auto penguins = DataFrame.penguins();
  double[] is_male;
  double[] body_mass;
  string[] island;
  auto n = penguins.nrows();
  for (size_t i = 0; i < n; i++) {
    if (penguins.is_null("sex", i) || penguins.is_null("body_mass_g", i) || penguins.is_null("island", i)) continue;
    is_male ~= (penguins.string_at("sex", i) == "male" ? 1.0 : 0.0);
    body_mass ~= penguins.numeric_at("body_mass_g", i);
    island ~= penguins.string_at("island", i);
  }

  auto sex_df = new DataFrame();
  sex_df.add_numeric_column("is_male", dv(is_male));
  sex_df.add_numeric_column("body_mass_g", dv(body_mass));
  sex_df.add_string_column("island", sv(island));

  auto sex_model = new GLMM(sex_df, "is_male ~ body_mass_g + (1 | island)", "binomial");
  sex_model.print_summary();

  writeln("\n=================== Poisson mixed model on simulated multi-site count data ===================");
  auto rng = Random(4242);
  int n_stores = 25;
  double[] store_effect;
  for (int s = 0; s < n_stores; s++) store_effect ~= gauss(rng) * 0.4;

  double true_intercept = 2.0;
  double true_slope = 0.3;
  double[] store;
  double[] promo;
  double[] visits;
  for (int s = 0; s < n_stores; s++) {
    int n_days = 15 + cast(int) uniform(0, 11, rng);
    for (int d = 0; d < n_days; d++) {
      double promo_intensity = uniform(0.0, 3.0, rng);
      double lam = exp(true_intercept + store_effect[s] + true_slope * promo_intensity);
      // Knuth's Poisson sampler.
      double l_thresh = exp(-lam);
      int k = 0;
      double p = 1.0;
      while (true) {
        k++;
        p *= uniform(0.0, 1.0, rng);
        if (p <= l_thresh) break;
      }
      store ~= cast(double) s;
      promo ~= promo_intensity;
      visits ~= cast(double)(k - 1);
    }
  }

  auto df = new DataFrame();
  df.add_numeric_column("store", dv(store));
  df.add_numeric_column("promo", dv(promo));
  df.add_numeric_column("visits", dv(visits));

  auto store_model = new GLMM(df, "visits ~ promo + (1 | store)", "poisson");
  store_model.print_summary();

  writeln("\nTrue generating values: intercept=", true_intercept, ", slope=", true_slope,
          ", random-intercept SD (log scale)=0.4");

  writeln("\n--- BLUPs for a few stores ---");
  auto group_labels = store_model.group_labels();
  foreach (idx; [0, 1, 2]) {
    writeln("store ", group_labels[idx], ": intercept shift=", store_model.random_effects_for_group(idx)[0]);
  }

  writeln("\n--- Prediction: population-level vs. store-adjusted ---");
  auto newdata_population = new DataFrame();
  newdata_population.add_numeric_column("promo", dv([1.5]));
  auto newdata_store0 = new DataFrame();
  newdata_store0.add_numeric_column("promo", dv([1.5]));
  newdata_store0.add_numeric_column("store", dv([0.0]));
  writeln("promo=1.5, unseen store:   ", store_model.predict(newdata_population)[0], " expected visits (fixed effects only)");
  writeln("promo=1.5, store 0 (known): ", store_model.predict(newdata_store0)[0], " expected visits (fixed effects + store 0's BLUP)");
}
