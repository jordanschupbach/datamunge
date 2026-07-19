using System;

class Program {
  static void Main() {
    var iris = DataFrame.iris();
    Console.WriteLine($"iris: {iris.nrows()} rows x {iris.ncols()} cols\n");

    var model = new XGBoostClassifier(iris, "Species ~ Petal.Length + Petal.Width");
    model.print_summary();

    Console.WriteLine("\nConfusion matrix (rows = actual, cols = predicted):");
    Console.WriteLine(model.confusion_matrix().to_string());

    Console.WriteLine("\nMisclassified rows:");
    var predictions = model.predict(iris);
    int misclassified = 0;
    uint n = iris.nrows();
    for (uint i = 0; i < n; i++) {
      var actual = iris.string_at("Species", i);
      var pred = predictions[(int)i];
      if (pred != actual) {
        misclassified += 1;
        Console.WriteLine($"  row {i}: Petal.Length={iris.numeric_at("Petal.Length", i)} Petal.Width={iris.numeric_at("Petal.Width", i)}  actual={actual}  predicted={pred}");
      }
    }
    Console.WriteLine($"{misclassified} of {n} misclassified ({(100.0 * misclassified) / n}%)");

    model.plot_training_deviance().save("xgboost_iris_training_deviance.svg");
    model.plot_decision_regions("Petal.Length", "Petal.Width").save("xgboost_iris_decision_regions.svg");
    Console.WriteLine("\nSaved xgboost_iris_training_deviance.svg and xgboost_iris_decision_regions.svg");

    var heavy = new XGBoostClassifier(iris, "Species ~ Petal.Length + Petal.Width", 100, 0.3, 6, 50.0);
    var modelDev = model.training_deviance();
    var heavyDev = heavy.training_deviance();
    Console.WriteLine($"\nlambda=1 (default):   training accuracy={model.training_accuracy() * 100.0}%  deviance={modelDev[modelDev.Count - 1]}");
    Console.WriteLine($"lambda=50 (heavy L2): training accuracy={heavy.training_accuracy() * 100.0}%  deviance={heavyDev[heavyDev.Count - 1]}");
    heavy.plot_decision_regions("Petal.Length", "Petal.Width").save("xgboost_iris_decision_regions_heavy_lambda.svg");
    Console.WriteLine("Saved xgboost_iris_decision_regions_heavy_lambda.svg");
  }
}
