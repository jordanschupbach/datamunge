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

  writeln("=== RBF kernel (default) ===");
  auto rbf_model = new SVM(iris, "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width");
  rbf_model.print_summary();
  writeln("\nConfusion matrix (rows = actual, cols = predicted):");
  writeln(rbf_model.confusion_matrix().to_string());

  writeln("\n=== Linear kernel, for comparison ===");
  auto linear_model = new SVM(iris, "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width", "linear");
  writeln("Training accuracy: ", linear_model.training_accuracy() * 100.0, "%");
  writeln("Support vectors: ", linear_model.num_support_vectors());

  auto newdata = new DataFrame();
  newdata.add_numeric_column("Sepal.Length", dv([5.1, 6.0, 6.5, 6.2]));
  newdata.add_numeric_column("Sepal.Width", dv([3.5, 2.7, 3.0, 2.8]));
  newdata.add_numeric_column("Petal.Length", dv([1.4, 4.5, 5.5, 4.8]));
  newdata.add_numeric_column("Petal.Width", dv([0.2, 1.5, 2.0, 1.8]));

  writeln("\nRBF predictions for new flowers (votes out of 3 one-vs-one pairs):");
  writeln(rbf_model.predict_frame(newdata).to_string());
}
