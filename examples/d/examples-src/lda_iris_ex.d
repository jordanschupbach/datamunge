module app;

import std.stdio : writeln;
import datamunge;

DVector dv(double[] t) {
  auto v = new DVector();
  foreach (x; t) v.push_back(x);
  return v;
}

void main() {
  auto iris = DataFrame.iris();
  writeln("iris: ", iris.nrows(), " rows x ", iris.ncols(), " cols\n");

  auto model = new LDA(iris, "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width");
  model.print_summary();

  writeln("\nConfusion matrix (rows = actual, cols = predicted):");
  writeln(model.confusion_matrix().to_string());

  auto newdata = new DataFrame();
  newdata.add_numeric_column("Sepal.Length", dv([5.1, 6.0, 6.5, 6.2]));
  newdata.add_numeric_column("Sepal.Width", dv([3.5, 2.7, 3.0, 2.8]));
  newdata.add_numeric_column("Petal.Length", dv([1.4, 4.5, 5.5, 4.8]));
  newdata.add_numeric_column("Petal.Width", dv([0.2, 1.5, 2.0, 1.8]));

  writeln("\nPredictions for new flowers:");
  writeln(model.predict_frame(newdata).to_string());

  model.save_discriminant_plot("lda_iris_discriminants.svg");
  writeln("\nSaved discriminant plot as lda_iris_discriminants.svg");
}
