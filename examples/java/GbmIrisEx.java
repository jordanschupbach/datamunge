// CMD: just TARGET=gbm_iris_ex run-java
// Gradient-boosted trees (GBM) on iris: summary, confusion matrix, misclassified rows, 5 vs 100 rounds.
import datamunge.GBMClassifier;
import datamunge.data.Data;
import java.util.List;

public class GbmIrisEx {
  static double last(double[] values) {
    return values[values.length - 1];
  }

  public static void main(String[] args) {
    String formula = "Species ~ Petal.Length + Petal.Width";

    var iris = Data.iris();
    System.out.println("iris: " + iris.nrows() + " rows x " + iris.ncols() + " cols\n");

    var model = new GBMClassifier(iris, formula);
    // summary() returns the text printSummary() writes to native stdout, so going through
    // System.out keeps it in order with the surrounding println calls.
    System.out.print(model.summary());

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
          "  row " + i + ": Petal.Length=" + iris.numericAt("Petal.Length", i)
              + " Petal.Width=" + iris.numericAt("Petal.Width", i)
              + "  actual=" + actual + "  predicted=" + predicted);
    }
    System.out.println(
        misclassified + " of " + n + " misclassified (" + (100.0 * misclassified / n) + "%)");

    model.plotTrainingDeviance().save("gbm_iris_training_deviance.svg");
    model.plotDecisionRegions("Petal.Length", "Petal.Width").save("gbm_iris_decision_regions.svg");
    System.out.println(
        "\nSaved gbm_iris_training_deviance.svg and gbm_iris_decision_regions.svg");

    // A handful of boosting rounds vs a well-boosted ensemble: each additional round chips away
    // at the training loss, gradually sharpening the decision boundary.
    // new GBMClassifier(data, formula, nTrees, learningRate, maxDepth, minSamplesSplit,
    //     minSamplesLeaf, subsample, seed)
    var few = new GBMClassifier(iris, formula, 5);
    System.out.println(
        "\n5-round ensemble:   training accuracy=" + (100 * few.trainingAccuracy())
            + "%  deviance=" + last(few.trainingDeviance()));
    System.out.println(
        "100-round ensemble: training accuracy=" + (100 * model.trainingAccuracy())
            + "%  deviance=" + last(model.trainingDeviance()));
    few.plotDecisionRegions("Petal.Length", "Petal.Width")
        .save("gbm_iris_decision_regions_5rounds.svg");
    System.out.println("Saved gbm_iris_decision_regions_5rounds.svg");
  }
}
