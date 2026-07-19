using System;

class Program {
  static void Main() {
    var iris = DataFrame.iris();
    Console.WriteLine($"iris: {iris.nrows()} rows x {iris.ncols()} cols\n");

    var model = new DecisionTreeClassifier(iris, "Species ~ Petal.Length + Petal.Width");
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

    model.plot_classification(iris, "Petal.Length", "Petal.Width").save("tree_iris_classification.svg");
    model.plot_decision_regions("Petal.Length", "Petal.Width").save("tree_iris_decision_regions.svg");
    Console.WriteLine("\nSaved tree_iris_classification.svg and tree_iris_decision_regions.svg");

    var shallow = new DecisionTreeClassifier(iris, "Species ~ Petal.Length + Petal.Width", 2);
    Console.WriteLine($"\nDepth-2 tree training accuracy: {shallow.training_accuracy() * 100.0}% ({shallow.leaf_count()} leaves)");
    shallow.plot_decision_regions("Petal.Length", "Petal.Width").save("tree_iris_decision_regions_depth2.svg");
    Console.WriteLine("Saved tree_iris_decision_regions_depth2.svg");
  }
}
