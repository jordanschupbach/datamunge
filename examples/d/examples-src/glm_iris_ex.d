module app;

import std.stdio : writeln;
import std.algorithm : minElement, maxElement, sum;
import std.math : round;
import datamunge;

DVector dv(double[] t) {
  auto v = new DVector();
  foreach (x; t) v.push_back(x);
  return v;
}

void main() {
  auto iris = DataFrame.iris();

  // Logistic regression: versicolor vs virginica only.
  double[] is_virginica;
  double[] petal_length;
  double[] petal_width;
  auto n = iris.nrows();
  for (size_t i = 0; i < n; i++) {
    auto species = iris.string_at("Species", i);
    if (species != "versicolor" && species != "virginica") continue;
    is_virginica ~= (species == "virginica" ? 1.0 : 0.0);
    petal_length ~= iris.numeric_at("Petal.Length", i);
    petal_width ~= iris.numeric_at("Petal.Width", i);
  }

  auto sub = new DataFrame();
  sub.add_numeric_column("Petal.Length", dv(petal_length));
  sub.add_numeric_column("Petal.Width", dv(petal_width));
  sub.add_numeric_column("is_virginica", dv(is_virginica));

  writeln("=================== Logistic regression (binomial, logit link) ===================");
  auto logit = new GLM(sub, "is_virginica ~ Petal.Length + Petal.Width", "binomial");
  logit.print_summary();

  auto fitted = logit.fitted_values();
  int correct = 0;
  for (size_t i = 0; i < is_virginica.length; i++) {
    if ((fitted[i] >= 0.5) == (is_virginica[i] >= 0.5)) correct++;
  }
  writeln("\nResubstitution accuracy at 0.5 threshold: ", 100.0 * correct / is_virginica.length, "%");

  logit.save_diagnostic_plots("glm_logistic_iris");
  writeln("\nSaved glm_logistic_iris_{residuals_vs_fitted,normal_qq,scale_location,residuals_vs_leverage}.svg");

  // Predicted-probability curve across Petal.Length, with Petal.Width held at its mean.
  double width_mean = sum(petal_width) / petal_width.length;
  int grid_n = 100;
  double pl_min = minElement(petal_length) - 0.3;
  double pl_max = maxElement(petal_length) + 0.3;
  double[] grid_x;
  double[] width_col;
  for (int i = 0; i < grid_n; i++) {
    grid_x ~= pl_min + (pl_max - pl_min) * i / (grid_n - 1);
    width_col ~= width_mean;
  }
  auto grid = new DataFrame();
  grid.add_numeric_column("Petal.Length", dv(grid_x));
  grid.add_numeric_column("Petal.Width", dv(width_col));
  auto curve_frame = logit.predict_frame(grid, "confidence");
  writeln("\nPredicted-probability curve (first 5 rows):");
  writeln(curve_frame.to_string(5));

  // Poisson regression, for contrast: same IRLS engine, different family/link.
  writeln("\n=================== Poisson regression (log link) ===================");
  double[] count;
  double[] sepal_width;
  double[] all_petal_length;
  for (size_t i = 0; i < n; i++) {
    count ~= round(iris.numeric_at("Sepal.Length", i));
    sepal_width ~= iris.numeric_at("Sepal.Width", i);
    all_petal_length ~= iris.numeric_at("Petal.Length", i);
  }
  auto count_data = new DataFrame();
  count_data.add_numeric_column("Sepal.Width", dv(sepal_width));
  count_data.add_numeric_column("Petal.Length", dv(all_petal_length));
  count_data.add_numeric_column("count", dv(count));

  auto poisson = new GLM(count_data, "count ~ Sepal.Width + Petal.Length", "poisson");
  poisson.print_summary();
}
