module app;

import std.stdio : writeln;
import datamunge;

void main() {
  auto iris = DataFrame.iris();
  writeln("iris: ", iris.nrows(), " rows x ", iris.ncols(), " cols\n");

  auto model = new RandomForestClassifier(iris, "Species ~ Petal.Length + Petal.Width");
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

  model.plot_classification(iris, "Petal.Length", "Petal.Width").save("forest_iris_classification.svg");
  model.plot_decision_regions("Petal.Length", "Petal.Width").save("forest_iris_decision_regions.svg");
  writeln("\nSaved forest_iris_classification.svg and forest_iris_decision_regions.svg");

  // A small forest, for comparison, showing how out-of-bag accuracy stabilizes as more trees
  // are added -- the defining random forest effect a single decision tree cannot demonstrate.
  auto small_forest = new RandomForestClassifier(iris, "Species ~ Petal.Length + Petal.Width", 5);
  writeln("\n5-tree forest:   training accuracy=", small_forest.training_accuracy() * 100.0,
          "%  OOB accuracy=", small_forest.oob_accuracy() * 100.0, "%");
  writeln("100-tree forest: training accuracy=", model.training_accuracy() * 100.0,
          "%  OOB accuracy=", model.oob_accuracy() * 100.0, "%");
  small_forest.plot_decision_regions("Petal.Length", "Petal.Width").save("forest_iris_decision_regions_5trees.svg");
  writeln("Saved forest_iris_decision_regions_5trees.svg");
}
