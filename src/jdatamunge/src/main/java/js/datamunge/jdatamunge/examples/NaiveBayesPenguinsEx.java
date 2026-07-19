package js.datamunge.jdatamunge.examples;

import js.datamunge.jdatamunge.DataFrame;
import js.datamunge.jdatamunge.NaiveBayesClassifier;

public class NaiveBayesPenguinsEx {
  static {
    System.loadLibrary("datamunge_jni");
  }

  public static void run() {
    var penguins = DataFrame.penguins();
    System.out.println("penguins: " + penguins.nrows() + " rows x " + penguins.ncols() + " cols\n");

    var model = new NaiveBayesClassifier(penguins, "species ~ bill_length_mm + bill_depth_mm + island + sex");
    model.print_summary();

    System.out.println("\nConfusion matrix (rows = actual, cols = predicted):");
    System.out.println(model.confusion_matrix().to_string());

    System.out.println("\nMisclassified rows:");
    var predictions = model.predict(penguins);
    int misclassified = 0;
    long n = penguins.nrows();
    for (long i = 0; i < n; i++) {
      if (!penguins.is_null("species", i) && !penguins.is_null("bill_length_mm", i) &&
          !penguins.is_null("bill_depth_mm", i) && !penguins.is_null("island", i) && !penguins.is_null("sex", i)) {
        String actual = penguins.string_at("species", i);
        String pred = predictions.get((int) i);
        if (!pred.equals(actual)) {
          misclassified++;
          System.out.println("  row " + i + ": bill_length=" + penguins.numeric_at("bill_length_mm", i) + " bill_depth=" + penguins.numeric_at("bill_depth_mm", i) + " island=" + penguins.string_at("island", i) + " sex=" + penguins.string_at("sex", i) + "  actual=" + actual + "  predicted=" + pred);
        }
      }
    }
    System.out.println(misclassified + " misclassified (of " + n + " rows, some incomplete)");

    var billOnly = new NaiveBayesClassifier(penguins, "species ~ bill_length_mm + bill_depth_mm");
    System.out.println("\nbill-measurements-only model training accuracy: " + (billOnly.training_accuracy() * 100.0) + "%");
    billOnly.plot_classification(penguins, "bill_length_mm", "bill_depth_mm").save("naive_bayes_penguins_classification.svg");
    billOnly.plot_decision_regions("bill_length_mm", "bill_depth_mm").save("naive_bayes_penguins_decision_regions.svg");
    System.out.println("Saved naive_bayes_penguins_classification.svg and naive_bayes_penguins_decision_regions.svg");
  }
}
