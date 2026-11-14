// CMD: just TARGET=xgboost_iris_ex run-java
// XGBoost classifier on iris: summary, confusion matrix, misclassified rows, L2-regularization comparison.
import datamunge.XGBoostClassifier;
import datamunge.data.Data;
import java.util.List;

public class XgboostIrisEx {
  static double last(double[] values) {
    return values[values.length - 1];
  }

  public static void main(String[] args) {
    var iris = Data.iris();
    System.out.println("iris: " + iris.nrows() + " rows x " + iris.ncols() + " cols\n");

    var model = new XGBoostClassifier(iris, "Species ~ Petal.Length + Petal.Width");
    model.printSummary();

    System.out.println(
        "\nConfusion matrix (rows = actual, cols = predicted):\n" + model.confusionMatrix());

    System.out.println("\nMisclassified rows:");
    List<String> predictions = model.predict(iris);
    long n = iris.nrows();
    int misclassified = 0;
    for (int i = 0; i < n; i++) {
      String actual = iris.stringAt("Species", i);
      String predicted = predictions.get(i);
      if (predicted.equals(actual)) {
        continue;
      }
      misclassified++;
      System.out.println(
          "  row " + i + ": Petal.Length=" + iris.numericAt("Petal.Length", i) + " Petal.Width="
              + iris.numericAt("Petal.Width", i) + "  actual=" + actual + "  predicted="
              + predicted);
    }
    System.out.println(
        misclassified + " of " + n + " misclassified (" + (100.0 * misclassified / n) + "%)");

    model.plotTrainingDeviance().save("xgboost_iris_training_deviance.svg");
    model
        .plotDecisionRegions("Petal.Length", "Petal.Width")
        .save("xgboost_iris_decision_regions.svg");
    System.out.println(
        "\nSaved xgboost_iris_training_deviance.svg and xgboost_iris_decision_regions.svg");

    // XGBoost's defining lever is regularization: a heavily L2-regularized model (large lambda)
    // grows the same deep trees but keeps every leaf weight small, producing a much smoother
    // decision boundary than the lightly-regularized default.
    // new XGBoostClassifier(data, formula, nTrees, learningRate, maxDepth, lambda, alpha, gamma,
    //     minChildWeight, minSamplesLeaf, subsample, colsampleBytree, seed); trailing args optional
    var heavy =
        new XGBoostClassifier(iris, "Species ~ Petal.Length + Petal.Width", 100, 0.3, 6, 50.0);
    System.out.println(
        "\nlambda=1 (default):   training accuracy=" + (100 * model.trainingAccuracy())
            + "%  deviance=" + last(model.trainingDeviance()));
    System.out.println(
        "lambda=50 (heavy L2): training accuracy=" + (100 * heavy.trainingAccuracy())
            + "%  deviance=" + last(heavy.trainingDeviance()));
    heavy
        .plotDecisionRegions("Petal.Length", "Petal.Width")
        .save("xgboost_iris_decision_regions_heavy_lambda.svg");
    System.out.println("Saved xgboost_iris_decision_regions_heavy_lambda.svg");
  }
}
