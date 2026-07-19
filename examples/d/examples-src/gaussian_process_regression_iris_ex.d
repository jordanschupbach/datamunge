module app;

import std.stdio : writeln;
import datamunge;

DVector dv(double[] t) {
  auto v = new DVector();
  foreach (x; t) v.push_back(x);
  return v;
}

void main() {
  string FORMULA = "Petal.Length ~ Petal.Width";

  auto iris = DataFrame.iris();
  writeln("iris: ", iris.nrows(), " rows x ", iris.ncols(), " cols");
  writeln("formula: ", FORMULA, "\n");

  // Both the length scale and the noise ratio are auto-selected by maximizing the exact log
  // marginal likelihood.
  auto model = new GaussianProcessRegression(iris, FORMULA);
  model.print_summary();

  model.plot_fit(iris).save("gpr_iris_fit.svg");
  model.plot_length_scale_profile().save("gpr_iris_length_scale_profile.svg");
  writeln("\nSaved gpr_iris_fit.svg and gpr_iris_length_scale_profile.svg");

  // Unlike every other regressor in this suite, a GP gives a genuine posterior confidence
  // interval at every point -- including far outside the training data.
  auto query = new DataFrame();
  query.add_numeric_column("Petal.Width", dv([0.2, 1.3, 2.5, 10.0]));
  auto detail = model.predict_frame(query, "confidence");
  writeln("\nPredictions with 95% confidence intervals:");
  writeln(detail.to_string());
  writeln("(Petal.Width=10.0 is far outside the training range [0.1, 2.5] -- note how much wider its\n" ~
          " interval is than the in-range predictions.)");
}
