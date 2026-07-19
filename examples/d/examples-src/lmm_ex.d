module app;

import std.stdio : writeln;
import std.random : Random, uniform, unpredictableSeed;
import std.math : sqrt, log, cos, PI;
import datamunge;

DVector dv(double[] t) {
  auto v = new DVector();
  foreach (x; t) v.push_back(x);
  return v;
}

double gauss(ref Random rng) {
  double u1 = uniform(0.0, 1.0, rng);
  double u2 = uniform(0.0, 1.0, rng);
  return sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
}

void main() {
  writeln("=================== Random intercept on a real dataset (penguins) ===================");
  auto penguins = DataFrame.penguins();
  auto species_model = new LMM(penguins, "body_mass_g ~ flipper_length_mm + bill_length_mm + (1 | species)");
  species_model.print_summary();

  writeln("\n=================== Random intercept + slope on a simulated multi-school dataset ===================");
  auto rng = Random(2024);
  int n_schools = 30;
  double[] school_intercept;
  double[] school_slope;
  for (int s = 0; s < n_schools; s++) {
    school_intercept ~= gauss(rng) * 6.0;
    school_slope ~= gauss(rng) * 1.2;
  }

  double true_intercept = 60.0;
  double true_slope = 3.0;
  double[] school;
  double[] study_hours;
  double[] score;
  for (int s = 0; s < n_schools; s++) {
    int n_students = 15 + cast(int) uniform(0, 21, rng);
    for (int j = 0; j < n_students; j++) {
      double hours = uniform(0.0, 10.0, rng);
      double noise = gauss(rng) * 4.0;
      double s_val = true_intercept + school_intercept[s] + (true_slope + school_slope[s]) * hours + noise;
      school ~= cast(double) s;
      study_hours ~= hours;
      score ~= s_val;
    }
  }

  auto df = new DataFrame();
  df.add_numeric_column("school", dv(school));
  df.add_numeric_column("study_hours", dv(study_hours));
  df.add_numeric_column("score", dv(score));

  auto model = new LMM(df, "score ~ study_hours + (1 + study_hours | school)");
  model.print_summary();

  writeln("\nTrue generating values: intercept=", true_intercept, ", slope=", true_slope,
          ", random-intercept SD=6.0, random-slope SD=1.2, residual SD=4.0");

  writeln("\n--- Best Linear Unbiased Predictors (BLUPs) for a few schools ---");
  auto group_labels = model.group_labels();
  foreach (idx; [0, 1, 2]) {
    auto re = model.random_effects_for_group(idx);
    writeln("school ", group_labels[idx], ": intercept shift=", re[0], ", slope shift=", re[1]);
  }

  writeln("\n--- Prediction: population-level vs. school-adjusted ---");
  auto newdata_population = new DataFrame();
  newdata_population.add_numeric_column("study_hours", dv([5.0]));
  auto newdata_school0 = new DataFrame();
  newdata_school0.add_numeric_column("study_hours", dv([5.0]));
  newdata_school0.add_numeric_column("school", dv([0.0]));
  writeln("5 study hours, unseen school:      ", model.predict(newdata_population)[0], " (fixed effects only)");
  writeln("5 study hours, school 0 (known):    ", model.predict(newdata_school0)[0], " (fixed effects + school 0's BLUP)");
}
