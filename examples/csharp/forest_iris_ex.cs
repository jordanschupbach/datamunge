using System;

class Program {
  static void Main() {
    var iris = DataFrame.iris();
    Console.WriteLine($"iris: {iris.nrows()} rows x {iris.ncols()} cols\n");

    var model = new RandomForestClassifier(iris, "Species ~ Petal.Length + Petal.Width");
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

    model.plot_classification(iris, "Petal.Length", "Petal.Width").save("forest_iris_classification.svg");
    model.plot_decision_regions("Petal.Length", "Petal.Width").save("forest_iris_decision_regions.svg");
    Console.WriteLine("\nSaved forest_iris_classification.svg and forest_iris_decision_regions.svg");

    var smallForest = new RandomForestClassifier(iris, "Species ~ Petal.Length + Petal.Width", 5);
    Console.WriteLine($"\n5-tree forest:   training accuracy={smallForest.training_accuracy() * 100.0}%  OOB accuracy={smallForest.oob_accuracy() * 100.0}%");
    Console.WriteLine($"100-tree forest: training accuracy={model.training_accuracy() * 100.0}%  OOB accuracy={model.oob_accuracy() * 100.0}%");
    smallForest.plot_decision_regions("Petal.Length", "Petal.Width").save("forest_iris_decision_regions_5trees.svg");
    Console.WriteLine("Saved forest_iris_decision_regions_5trees.svg");
  }
}
