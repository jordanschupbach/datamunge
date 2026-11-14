// CMD: just TARGET=naive_bayes_penguins_ex run-java
// Naive Bayes on penguins: mixed numeric/categorical predictors, confusion matrix, misclassified rows, plots.
import datamunge.NaiveBayesClassifier;
import datamunge.data.Data;
import java.util.List;

public class NaiveBayesPenguinsEx {
  public static void main(String[] args) {
    // Note: NaiveBayesClassifier drops incomplete rows for its own formula columns internally
    // when fitting, matching every other formula-based model in this library.
    var penguins = Data.penguins();
    System.out.println("penguins: " + penguins.nrows() + " rows x " + penguins.ncols() + " cols\n");

    // A mix of numeric (Gaussian likelihood) and categorical (frequency-table likelihood)
    // predictors in one formula -- each modeled independently given the class, per the naive
    // Bayes assumption.
    var model =
        new NaiveBayesClassifier(
            penguins, "species ~ bill_length_mm + bill_depth_mm + island + sex");
    model.printSummary();
    System.out.flush();

    System.out.println(
        "\nConfusion matrix (rows = actual, cols = predicted):\n" + model.confusionMatrix());

    System.out.println("\nMisclassified rows:");
    List<String> predictions = model.predict(penguins);
    int misclassified = 0;
    List<String> columns = List.of("species", "bill_length_mm", "bill_depth_mm", "island", "sex");
    for (int i = 0; i < penguins.nrows(); i++) {
      final long row = i;
      if (columns.stream().anyMatch(column -> penguins.isNull(column, row))) {
        continue;
      }
      String actual = penguins.stringAt("species", i);
      String predicted = predictions.get(i);
      if (predicted.equals(actual)) {
        continue;
      }
      misclassified++;
      System.out.println(
          "  row " + i
              + ": bill_length=" + penguins.numericAt("bill_length_mm", i)
              + " bill_depth=" + penguins.numericAt("bill_depth_mm", i)
              + " island=" + penguins.stringAt("island", i)
              + " sex=" + penguins.stringAt("sex", i)
              + "  actual=" + actual
              + "  predicted=" + predicted);
    }
    System.out.println(
        misclassified + " misclassified (of " + penguins.nrows() + " rows, some incomplete)");

    // plot_decision_regions requires exactly two NUMERIC predictors, so build a separate
    // two-predictor model (bill measurements alone) just for visualization.
    var billOnly = new NaiveBayesClassifier(penguins, "species ~ bill_length_mm + bill_depth_mm");
    System.out.println(
        "\nbill-measurements-only model training accuracy: "
            + 100 * billOnly.trainingAccuracy()
            + "%");
    billOnly
        .plotClassification(penguins, "bill_length_mm", "bill_depth_mm")
        .save("naive_bayes_penguins_classification.svg");
    billOnly
        .plotDecisionRegions("bill_length_mm", "bill_depth_mm")
        .save("naive_bayes_penguins_decision_regions.svg");
    System.out.println(
        "Saved naive_bayes_penguins_classification.svg and"
            + " naive_bayes_penguins_decision_regions.svg");
  }
}
