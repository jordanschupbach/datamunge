// CMD: just TARGET=knn_iris_ex run-java
// k-nearest-neighbours classifier on iris: leave-one-out accuracy and decision regions for several k.
import datamunge.KNNClassifier;
import datamunge.data.Data;
import java.util.List;
import java.util.Locale;

public class KnnIrisEx {
  public static void main(String[] args) {
    String formula = "Species ~ Petal.Length + Petal.Width";

    var iris = Data.iris();
    System.out.println("iris: " + iris.nrows() + " rows x " + iris.ncols() + " cols\n");

    var model = new KNNClassifier(iris, formula);
    // summary() returns the text printSummary() writes to native stdout, so going through
    // System.out keeps it in order with the surrounding println calls.
    System.out.print(model.summary());

    System.out.println("\nConfusion matrix (leave-one-out; rows = actual, cols = predicted):");
    System.out.println(model.confusionMatrix());

    System.out.println("\nLeave-one-out misclassified rows:");
    List<String> fitted = model.predict(iris);
    long n = iris.nrows();
    int misclassified = 0;
    for (int i = 0; i < n; i++) {
      String actual = iris.stringAt("Species", i);
      String predicted = fitted.get(i);
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

    model
        .plotDecisionRegions("Petal.Length", "Petal.Width")
        .save("knn_iris_decision_regions_k5.svg");
    System.out.println("\nSaved knn_iris_decision_regions_k5.svg");

    // k=1 memorizes every training point exactly (jagged, overfit boundary with an island around
    // every point, including noise); k=25 averages over a much larger neighborhood (very smooth,
    // underfit boundary). new KNNClassifier(data, formula, k, metric, weighted, standardize)
    for (long k : new long[] {1, 25}) {
      var knn = new KNNClassifier(iris, formula, k);
      String file = "knn_iris_decision_regions_k" + k + ".svg";
      System.out.println(
          "\nk=" + String.format(Locale.ROOT, "%-2s", k) + " leave-one-out accuracy: "
              + (knn.trainingAccuracy() * 100) + "%");
      knn.plotDecisionRegions("Petal.Length", "Petal.Width").save(file);
      System.out.println("Saved " + file);
    }
  }
}
