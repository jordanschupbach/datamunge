module app;

import std.stdio : writeln;
import datamunge;

void main() {
  auto iris = DataFrame.iris();
  writeln("iris: ", iris.nrows(), " rows x ", iris.ncols(), " cols\n");

  auto model = new KNNClassifier(iris, "Species ~ Petal.Length + Petal.Width");
  model.print_summary();

  writeln("\nConfusion matrix (leave-one-out; rows = actual, cols = predicted):");
  writeln(model.confusion_matrix().to_string());

  writeln("\nLeave-one-out misclassified rows:");
  auto fitted = model.predict(iris);
  int misclassified = 0;
  auto n = iris.nrows();
  for (size_t i = 0; i < n; i++) {
    auto actual = iris.string_at("Species", i);
    if (fitted[i] != actual) {
      misclassified++;
      writeln("  row ", i, ": Petal.Length=", iris.numeric_at("Petal.Length", i),
              " Petal.Width=", iris.numeric_at("Petal.Width", i),
              "  actual=", actual, "  predicted=", fitted[i]);
    }
  }
  writeln(misclassified, " of ", n, " misclassified (", 100.0 * misclassified / n, "%)");

  model.plot_decision_regions("Petal.Length", "Petal.Width").save("knn_iris_decision_regions_k5.svg");
  writeln("\nSaved knn_iris_decision_regions_k5.svg");

  // k=1 memorizes every training point exactly; k=25 averages over a much larger neighborhood.
  auto k1 = new KNNClassifier(iris, "Species ~ Petal.Length + Petal.Width", 1);
  writeln("\nk=1  leave-one-out accuracy: ", k1.training_accuracy() * 100.0, "%");
  k1.plot_decision_regions("Petal.Length", "Petal.Width").save("knn_iris_decision_regions_k1.svg");
  writeln("Saved knn_iris_decision_regions_k1.svg");

  auto k25 = new KNNClassifier(iris, "Species ~ Petal.Length + Petal.Width", 25);
  writeln("\nk=25 leave-one-out accuracy: ", k25.training_accuracy() * 100.0, "%");
  k25.plot_decision_regions("Petal.Length", "Petal.Width").save("knn_iris_decision_regions_k25.svg");
  writeln("Saved knn_iris_decision_regions_k25.svg");
}
