package js.datamunge.jdatamunge.examples;

import js.datamunge.jdatamunge.DataFrame;
import js.datamunge.jdatamunge.XGBoostClassifier;

public class XgboostIrisEx {
  static {
    System.loadLibrary("datamunge_jni");
  }

  public static void run() {
    var iris = DataFrame.iris();
    System.out.println("iris: " + iris.nrows() + " rows x " + iris.ncols() + " cols\n");

    var model = new XGBoostClassifier(iris, "Species ~ Petal.Length + Petal.Width");
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

    model.plot_training_deviance().save("xgboost_iris_training_deviance.svg");
    model.plot_decision_regions("Petal.Length", "Petal.Width").save("xgboost_iris_decision_regions.svg");
    System.out.println("\nSaved xgboost_iris_training_deviance.svg and xgboost_iris_decision_regions.svg");

    var heavy = new XGBoostClassifier(iris, "Species ~ Petal.Length + Petal.Width", 100, 0.3, 6, 50.0);
    var modelDev = model.training_deviance();
    var heavyDev = heavy.training_deviance();
    System.out.println("\nlambda=1 (default):   training accuracy=" + (model.training_accuracy() * 100.0) + "%  deviance=" + modelDev.get(modelDev.size() - 1));
    System.out.println("lambda=50 (heavy L2): training accuracy=" + (heavy.training_accuracy() * 100.0) + "%  deviance=" + heavyDev.get(heavyDev.size() - 1));
    heavy.plot_decision_regions("Petal.Length", "Petal.Width").save("xgboost_iris_decision_regions_heavy_lambda.svg");
    System.out.println("Saved xgboost_iris_decision_regions_heavy_lambda.svg");
  }
}
