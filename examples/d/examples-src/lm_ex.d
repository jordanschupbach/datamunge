module app;

import std.stdio : writeln;
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

void main() {
  double[] hp = [110.0, 110.0, 93.0, 110.0, 175.0, 105.0, 245.0, 62.0, 95.0, 123.0];
  double[] wt = [2.62, 2.875, 2.32, 3.215, 3.44, 3.46, 3.57, 3.19, 3.15, 3.44];
  string[] transmission = ["manual", "manual", "manual", "automatic", "automatic",
                            "automatic", "automatic", "automatic", "automatic", "automatic"];
  double[] mpg = [21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2];

  auto cars = new DataFrame();
  cars.add_numeric_column("hp", dv(hp));
  cars.add_numeric_column("wt", dv(wt));
  cars.add_string_column("transmission", sv(transmission));
  cars.add_numeric_column("mpg", dv(mpg));

  writeln("Fitting: mpg ~ hp + wt + transmission\n");
  auto model = new LM(cars, "mpg ~ hp + wt + transmission");
  model.print_summary();

  writeln("\nSequential ANOVA:");
  writeln(model.anova().to_string());

  auto newcars = new DataFrame();
  newcars.add_numeric_column("hp", dv([150.0, 90.0]));
  newcars.add_numeric_column("wt", dv([3.0, 2.5]));
  newcars.add_string_column("transmission", sv(["manual", "automatic"]));

  auto frame = model.predict_frame(newcars, "confidence");
  writeln("\nPredictions with 95% confidence intervals:");
  writeln(frame.to_string());

  model.save_diagnostic_plots("lm_ex_diagnostics");
  writeln("\nSaved diagnostic plots as lm_ex_diagnostics_*.svg");
}
