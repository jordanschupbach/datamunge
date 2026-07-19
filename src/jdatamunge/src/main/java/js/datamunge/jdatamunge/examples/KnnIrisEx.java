package js.datamunge.jdatamunge.examples;

import js.datamunge.jdatamunge.DataFrame;
import js.datamunge.jdatamunge.KNNClassifier;

public class KnnIrisEx {
  static {
    System.loadLibrary("datamunge_jni");
  }

  public static void run() {
    var iris = DataFrame.iris();
    System.out.println("iris: " + iris.nrows() + " rows x " + iris.ncols() + " cols\n");

    var model = new KNNClassifier(iris, "Species ~ Petal.Length + Petal.Width");
    model.print_summary();

    System.out.println("\nConfusion matrix (leave-one-out; rows = actual, cols = predicted):");
    System.out.println(model.confusion_matrix().to_string());

    System.out.println("\nLeave-one-out misclassified rows:");
    var fitted = model.predict(iris);
    int misclassified = 0;
    long n = iris.nrows();
    for (long i = 0; i < n; i++) {
      String actual = iris.string_at("Species", i);
      String pred = fitted.get((int) i);
      if (!pred.equals(actual)) {
        misclassified++;
        System.out.println("  row " + i + ": Petal.Length=" + iris.numeric_at("Petal.Length", i) + " Petal.Width=" + iris.numeric_at("Petal.Width", i) + "  actual=" + actual + "  predicted=" + pred);
      }
    }
    System.out.println(misclassified + " of " + n + " misclassified (" + (100.0 * misclassified / n) + "%)");

    model.plot_decision_regions("Petal.Length", "Petal.Width").save("knn_iris_decision_regions_k5.svg");
    System.out.println("\nSaved knn_iris_decision_regions_k5.svg");

    var k1 = new KNNClassifier(iris, "Species ~ Petal.Length + Petal.Width", 1);
    System.out.println("\nk=1  leave-one-out accuracy: " + (k1.training_accuracy() * 100.0) + "%");
    k1.plot_decision_regions("Petal.Length", "Petal.Width").save("knn_iris_decision_regions_k1.svg");
    System.out.println("Saved knn_iris_decision_regions_k1.svg");

    var k25 = new KNNClassifier(iris, "Species ~ Petal.Length + Petal.Width", 25);
    System.out.println("\nk=25 leave-one-out accuracy: " + (k25.training_accuracy() * 100.0) + "%");
    k25.plot_decision_regions("Petal.Length", "Petal.Width").save("knn_iris_decision_regions_k25.svg");
    System.out.println("Saved knn_iris_decision_regions_k25.svg");
  }
}
