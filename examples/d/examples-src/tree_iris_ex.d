module app;

import std.stdio : writeln;
import datamunge;

void main() {
  auto iris = DataFrame.iris();
  writeln("iris: ", iris.nrows(), " rows x ", iris.ncols(), " cols\n");

  auto model = new DecisionTreeClassifier(iris, "Species ~ Petal.Length + Petal.Width");
  model.print_summary();

  writeln("\nConfusion matrix (rows = actual, cols = predicted):");
  writeln(model.confusion_matrix().to_string());

  writeln("\nMisclassified rows:");
  auto predictions = model.predict(iris);
  int misclassified = 0;
  auto n = iris.nrows();
  for (size_t i = 0; i < n; i++) {
    auto actual = iris.string_at("Species", i);
    if (predictions[i] != actual) {
      misclassified++;
      writeln("  row ", i, ": Petal.Length=", iris.numeric_at("Petal.Length", i),
              " Petal.Width=", iris.numeric_at("Petal.Width", i),
              "  actual=", actual, "  predicted=", predictions[i]);
    }
  }
  writeln(misclassified, " of ", n, " misclassified (", 100.0 * misclassified / n, "%)");

  model.plot_classification(iris, "Petal.Length", "Petal.Width").save("tree_iris_classification.svg");
  model.plot_decision_regions("Petal.Length", "Petal.Width").save("tree_iris_decision_regions.svg");
  writeln("\nSaved tree_iris_classification.svg and tree_iris_decision_regions.svg");

  // A shallower tree, for comparison, showing a coarser (but still fairly accurate) decision boundary.
  auto shallow = new DecisionTreeClassifier(iris, "Species ~ Petal.Length + Petal.Width", 2);
  writeln("\nDepth-2 tree training accuracy: ", shallow.training_accuracy() * 100.0, "% (", shallow.leaf_count(), " leaves)");
  shallow.plot_decision_regions("Petal.Length", "Petal.Width").save("tree_iris_decision_regions_depth2.svg");
  writeln("Saved tree_iris_decision_regions_depth2.svg");
}
