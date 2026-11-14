// CMD: just TARGET=tree_iris_ex run-java
// Decision tree classifier on iris: summary, confusion matrix, misclassified rows and plots.
import datamunge.DecisionTreeClassifier;
import datamunge.data.Data;
import java.util.List;

public class TreeIrisEx {
  public static void main(String[] args) {
    var iris = Data.iris();
    System.out.println("iris: " + iris.nrows() + " rows x " + iris.ncols() + " cols\n");

    var model = new DecisionTreeClassifier(iris, "Species ~ Petal.Length + Petal.Width");
    model.printSummary();

    System.out.println("\nConfusion matrix (rows = actual, cols = predicted):");
    System.out.println(model.confusionMatrix());

    System.out.println("\nMisclassified rows:");
    List<String> predictions = model.predict(iris);
    long n = iris.nrows();
    int misclassified = 0;
    for (int i = 0; i < predictions.size(); i++) {
      String predicted = predictions.get(i);
      String actual = iris.stringAt("Species", i);
      if (predicted.equals(actual)) {
        continue;
      }
      misclassified++;
      System.out.println(
          "  row "
              + i
              + ": Petal.Length="
              + iris.numericAt("Petal.Length", i)
              + " Petal.Width="
              + iris.numericAt("Petal.Width", i)
              + "  actual="
              + actual
              + "  predicted="
              + predicted);
    }
    System.out.println(
        misclassified + " of " + n + " misclassified (" + 100.0 * misclassified / n + "%)");

    model
        .plotClassification(iris, "Petal.Length", "Petal.Width")
        .save("tree_iris_classification.svg");
    model.plotDecisionRegions("Petal.Length", "Petal.Width").save("tree_iris_decision_regions.svg");
    System.out.println("\nSaved tree_iris_classification.svg and tree_iris_decision_regions.svg");

    // A shallower tree, for comparison, showing a coarser (but still fairly accurate) decision
    // boundary.
    var shallow = new DecisionTreeClassifier(iris, "Species ~ Petal.Length + Petal.Width", 2);
    System.out.println(
        "\nDepth-2 tree training accuracy: "
            + shallow.trainingAccuracy() * 100
            + "% ("
            + shallow.leafCount()
            + " leaves)");
    shallow
        .plotDecisionRegions("Petal.Length", "Petal.Width")
        .save("tree_iris_decision_regions_depth2.svg");
    System.out.println("Saved tree_iris_decision_regions_depth2.svg");
  }
}
