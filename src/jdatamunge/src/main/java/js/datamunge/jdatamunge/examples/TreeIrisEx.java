package js.datamunge.jdatamunge.examples;

import js.datamunge.jdatamunge.DataFrame;
import js.datamunge.jdatamunge.DecisionTreeClassifier;

public class TreeIrisEx {
  static {
    System.loadLibrary("datamunge_jni");
  }

  public static void run() {
    var iris = DataFrame.iris();
    System.out.println("iris: " + iris.nrows() + " rows x " + iris.ncols() + " cols\n");

    var model = new DecisionTreeClassifier(iris, "Species ~ Petal.Length + Petal.Width");
    model.print_summary();

    System.out.println("\nConfusion matrix (rows = actual, cols = predicted):");
    System.out.println(model.confusion_matrix().to_string());

    System.out.println("\nMisclassified rows:");
    var predictions = model.predict(iris);
    int misclassified = 0;
    long n = iris.nrows();
    for (long i = 0; i < n; i++) {
      String actual = iris.string_at("Species", i);
      String pred = predictions.get((int) i);
      if (!pred.equals(actual)) {
        misclassified++;
        System.out.println("  row " + i + ": Petal.Length=" + iris.numeric_at("Petal.Length", i) + " Petal.Width=" + iris.numeric_at("Petal.Width", i) + "  actual=" + actual + "  predicted=" + pred);
      }
    }
    System.out.println(misclassified + " of " + n + " misclassified (" + (100.0 * misclassified / n) + "%)");

    model.plot_classification(iris, "Petal.Length", "Petal.Width").save("tree_iris_classification.svg");
    model.plot_decision_regions("Petal.Length", "Petal.Width").save("tree_iris_decision_regions.svg");
    System.out.println("\nSaved tree_iris_classification.svg and tree_iris_decision_regions.svg");

    var shallow = new DecisionTreeClassifier(iris, "Species ~ Petal.Length + Petal.Width", 2);
    System.out.println("\nDepth-2 tree training accuracy: " + (shallow.training_accuracy() * 100.0) + "% (" + shallow.leaf_count() + " leaves)");
    shallow.plot_decision_regions("Petal.Length", "Petal.Width").save("tree_iris_decision_regions_depth2.svg");
    System.out.println("Saved tree_iris_decision_regions_depth2.svg");
  }
}
