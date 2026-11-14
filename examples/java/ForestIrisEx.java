// CMD: just TARGET=forest_iris_ex run-java
// Random forest classifier on iris: summary, confusion matrix, misclassifications, OOB accuracy.
import datamunge.RandomForestClassifier;
import datamunge.data.Data;
import java.util.List;

public class ForestIrisEx {
  public static void main(String[] args) {
    var iris = Data.iris();
    System.out.println("iris: " + iris.nrows() + " rows x " + iris.ncols() + " cols\n");

    var model = new RandomForestClassifier(iris, "Species ~ Petal.Length + Petal.Width");
    model.printSummary();
    System.out.flush();

    System.out.println("\nConfusion matrix (rows = actual, cols = predicted):");
    System.out.println(model.confusionMatrix());

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
          "  row " + i
              + ": Petal.Length=" + iris.numericAt("Petal.Length", i)
              + " Petal.Width=" + iris.numericAt("Petal.Width", i)
              + "  actual=" + actual
              + "  predicted=" + predicted);
    }
    System.out.println(
        misclassified + " of " + n + " misclassified (" + 100.0 * misclassified / n + "%)");

    model
        .plotClassification(iris, "Petal.Length", "Petal.Width")
        .save("forest_iris_classification.svg");
    model.plotDecisionRegions("Petal.Length", "Petal.Width").save("forest_iris_decision_regions.svg");
    System.out.println("\nSaved forest_iris_classification.svg and forest_iris_decision_regions.svg");

    // A small forest, for comparison, showing how out-of-bag accuracy stabilizes as more trees
    // are added -- the defining random forest effect a single decision tree cannot demonstrate.
    // new RandomForestClassifier(data, formula, nTrees, maxDepth, minSamplesSplit, minSamplesLeaf,
    //     maxFeatures, criterion, bootstrap, sampleFraction, seed), trailing arguments optional.
    var smallForest = new RandomForestClassifier(iris, "Species ~ Petal.Length + Petal.Width", 5);
    System.out.println(
        "\n5-tree forest:   training accuracy=" + smallForest.trainingAccuracy() * 100
            + "%  OOB accuracy=" + smallForest.oobAccuracy() * 100 + "%");
    System.out.println(
        "100-tree forest: training accuracy=" + model.trainingAccuracy() * 100
            + "%  OOB accuracy=" + model.oobAccuracy() * 100 + "%");
    smallForest
        .plotDecisionRegions("Petal.Length", "Petal.Width")
        .save("forest_iris_decision_regions_5trees.svg");
    System.out.println("Saved forest_iris_decision_regions_5trees.svg");
  }
}
