module app;

import std.stdio : writeln;
import datamunge;

void main() {
  auto iris = DataFrame.iris();
  writeln("iris: ", iris.nrows(), " rows x ", iris.ncols(), " cols\n");

  auto model = new XGBoostClassifier(iris, "Species ~ Petal.Length + Petal.Width");
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

  model.plot_training_deviance().save("xgboost_iris_training_deviance.svg");
  model.plot_decision_regions("Petal.Length", "Petal.Width").save("xgboost_iris_decision_regions.svg");
  writeln("\nSaved xgboost_iris_training_deviance.svg and xgboost_iris_decision_regions.svg");

  // Heavy L2 regularization vs default.
  auto heavy = new XGBoostClassifier(iris, "Species ~ Petal.Length + Petal.Width", 100, 0.3, 6, 50.0);
  auto model_dev = model.training_deviance();
  auto heavy_dev = heavy.training_deviance();
  writeln("\nlambda=1 (default):   training accuracy=", model.training_accuracy() * 100.0, "%  deviance=", model_dev[$ - 1]);
  writeln("lambda=50 (heavy L2): training accuracy=", heavy.training_accuracy() * 100.0, "%  deviance=", heavy_dev[$ - 1]);
  heavy.plot_decision_regions("Petal.Length", "Petal.Width").save("xgboost_iris_decision_regions_heavy_lambda.svg");
  writeln("Saved xgboost_iris_decision_regions_heavy_lambda.svg");
}
