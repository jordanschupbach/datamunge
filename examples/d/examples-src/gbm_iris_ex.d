module app;

import std.stdio : writeln;
import datamunge;

void main() {
  auto iris = DataFrame.iris();
  writeln("iris: ", iris.nrows(), " rows x ", iris.ncols(), " cols\n");

  auto model = new GBMClassifier(iris, "Species ~ Petal.Length + Petal.Width");
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

  model.plot_training_deviance().save("gbm_iris_training_deviance.svg");
  model.plot_decision_regions("Petal.Length", "Petal.Width").save("gbm_iris_decision_regions.svg");
  writeln("\nSaved gbm_iris_training_deviance.svg and gbm_iris_decision_regions.svg");

  // A handful of boosting rounds vs a well-boosted ensemble.
  auto few = new GBMClassifier(iris, "Species ~ Petal.Length + Petal.Width", 5);
  auto few_dev = few.training_deviance();
  auto model_dev = model.training_deviance();
  writeln("\n5-round ensemble:   training accuracy=", few.training_accuracy() * 100.0, "%  deviance=", few_dev[$ - 1]);
  writeln("100-round ensemble: training accuracy=", model.training_accuracy() * 100.0, "%  deviance=", model_dev[$ - 1]);
  few.plot_decision_regions("Petal.Length", "Petal.Width").save("gbm_iris_decision_regions_5rounds.svg");
  writeln("Saved gbm_iris_decision_regions_5rounds.svg");
}
