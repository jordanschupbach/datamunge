using System;

class Program {
  static void Main() {
    var iris = DataFrame.iris();
    Console.WriteLine($"iris: {iris.nrows()} rows x {iris.ncols()} cols\n");

    var model = new KNNClassifier(iris, "Species ~ Petal.Length + Petal.Width");
    model.print_summary();

    Console.WriteLine("\nConfusion matrix (leave-one-out; rows = actual, cols = predicted):");
    Console.WriteLine(model.confusion_matrix().to_string());

    Console.WriteLine("\nLeave-one-out misclassified rows:");
    var fitted = model.predict(iris);
    int misclassified = 0;
    uint n = iris.nrows();
    for (uint i = 0; i < n; i++) {
      var actual = iris.string_at("Species", i);
      var pred = fitted[(int)i];
      if (pred != actual) {
        misclassified += 1;
        Console.WriteLine($"  row {i}: Petal.Length={iris.numeric_at("Petal.Length", i)} Petal.Width={iris.numeric_at("Petal.Width", i)}  actual={actual}  predicted={pred}");
      }
    }
    Console.WriteLine($"{misclassified} of {n} misclassified ({(100.0 * misclassified) / n}%)");

    model.plot_decision_regions("Petal.Length", "Petal.Width").save("knn_iris_decision_regions_k5.svg");
    Console.WriteLine("\nSaved knn_iris_decision_regions_k5.svg");

    var k1 = new KNNClassifier(iris, "Species ~ Petal.Length + Petal.Width", 1);
    Console.WriteLine($"\nk=1  leave-one-out accuracy: {k1.training_accuracy() * 100.0}%");
    k1.plot_decision_regions("Petal.Length", "Petal.Width").save("knn_iris_decision_regions_k1.svg");
    Console.WriteLine("Saved knn_iris_decision_regions_k1.svg");

    var k25 = new KNNClassifier(iris, "Species ~ Petal.Length + Petal.Width", 25);
    Console.WriteLine($"\nk=25 leave-one-out accuracy: {k25.training_accuracy() * 100.0}%");
    k25.plot_decision_regions("Petal.Length", "Petal.Width").save("knn_iris_decision_regions_k25.svg");
    Console.WriteLine("Saved knn_iris_decision_regions_k25.svg");
  }
}
