using System;

class Program {
  static void Main() {
    var iris = DataFrame.iris();
    Console.WriteLine($"iris: {iris.nrows()} rows x {iris.ncols()} cols\n");

    var model = new GBMClassifier(iris, "Species ~ Petal.Length + Petal.Width");
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

    model.plot_training_deviance().save("gbm_iris_training_deviance.svg");
    model.plot_decision_regions("Petal.Length", "Petal.Width").save("gbm_iris_decision_regions.svg");
    Console.WriteLine("\nSaved gbm_iris_training_deviance.svg and gbm_iris_decision_regions.svg");

    var few = new GBMClassifier(iris, "Species ~ Petal.Length + Petal.Width", 5);
    var fewDev = few.training_deviance();
    var modelDev = model.training_deviance();
    Console.WriteLine($"\n5-round ensemble:   training accuracy={few.training_accuracy() * 100.0}%  deviance={fewDev[fewDev.Count - 1]}");
    Console.WriteLine($"100-round ensemble: training accuracy={model.training_accuracy() * 100.0}%  deviance={modelDev[modelDev.Count - 1]}");
    few.plot_decision_regions("Petal.Length", "Petal.Width").save("gbm_iris_decision_regions_5rounds.svg");
    Console.WriteLine("Saved gbm_iris_decision_regions_5rounds.svg");
  }
}
