package js.datamunge.jdatamunge.examples;

import js.datamunge.jdatamunge.DataFrame;
import js.datamunge.jdatamunge.GBMClassifier;

public class GbmIrisEx {
  static {
    System.loadLibrary("datamunge_jni");
  }

  public static void run() {
    var iris = DataFrame.iris();
    System.out.println("iris: " + iris.nrows() + " rows x " + iris.ncols() + " cols\n");

    var model = new GBMClassifier(iris, "Species ~ Petal.Length + Petal.Width");
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

    model.plot_training_deviance().save("gbm_iris_training_deviance.svg");
    model.plot_decision_regions("Petal.Length", "Petal.Width").save("gbm_iris_decision_regions.svg");
    System.out.println("\nSaved gbm_iris_training_deviance.svg and gbm_iris_decision_regions.svg");

    var few = new GBMClassifier(iris, "Species ~ Petal.Length + Petal.Width", 5);
    var fewDev = few.training_deviance();
    var modelDev = model.training_deviance();
    System.out.println("\n5-round ensemble:   training accuracy=" + (few.training_accuracy() * 100.0) + "%  deviance=" + fewDev.get(fewDev.size() - 1));
    System.out.println("100-round ensemble: training accuracy=" + (model.training_accuracy() * 100.0) + "%  deviance=" + modelDev.get(modelDev.size() - 1));
    few.plot_decision_regions("Petal.Length", "Petal.Width").save("gbm_iris_decision_regions_5rounds.svg");
    System.out.println("Saved gbm_iris_decision_regions_5rounds.svg");
  }
}
