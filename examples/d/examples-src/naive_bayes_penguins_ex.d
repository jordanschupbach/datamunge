module app;

import std.stdio : writeln;
import datamunge;

void main() {
  // Note: unlike the C++ example, this doesn't call drop_nulls() first (that method isn't
  // exposed on the SWIG-bound DataFrame facade) -- NaiveBayesClassifier drops incomplete rows
  // for its own formula columns internally when fitting.
  auto penguins = DataFrame.penguins();
  writeln("penguins: ", penguins.nrows(), " rows x ", penguins.ncols(), " cols\n");

  auto model = new NaiveBayesClassifier(penguins, "species ~ bill_length_mm + bill_depth_mm + island + sex");
  model.print_summary();

  writeln("\nConfusion matrix (rows = actual, cols = predicted):");
  writeln(model.confusion_matrix().to_string());

  writeln("\nMisclassified rows:");
  auto predictions = model.predict(penguins);
  int misclassified = 0;
  auto n = penguins.nrows();
  for (size_t i = 0; i < n; i++) {
    if (penguins.is_null("species", i) || penguins.is_null("bill_length_mm", i) ||
        penguins.is_null("bill_depth_mm", i) || penguins.is_null("island", i) || penguins.is_null("sex", i))
      continue;
    auto actual = penguins.string_at("species", i);
    if (predictions[i] != actual) {
      misclassified++;
      writeln("  row ", i, ": bill_length=", penguins.numeric_at("bill_length_mm", i),
              " bill_depth=", penguins.numeric_at("bill_depth_mm", i),
              " island=", penguins.string_at("island", i),
              " sex=", penguins.string_at("sex", i),
              "  actual=", actual, "  predicted=", predictions[i]);
    }
  }
  writeln(misclassified, " misclassified (of ", n, " rows, some incomplete)");

  // plot_decision_regions requires exactly two NUMERIC predictors, so build a separate
  // two-predictor model (bill measurements alone) just for visualization.
  auto bill_only = new NaiveBayesClassifier(penguins, "species ~ bill_length_mm + bill_depth_mm");
  writeln("\nbill-measurements-only model training accuracy: ", bill_only.training_accuracy() * 100.0, "%");
  bill_only.plot_classification(penguins, "bill_length_mm", "bill_depth_mm").save("naive_bayes_penguins_classification.svg");
  bill_only.plot_decision_regions("bill_length_mm", "bill_depth_mm").save("naive_bayes_penguins_decision_regions.svg");
  writeln("Saved naive_bayes_penguins_classification.svg and naive_bayes_penguins_decision_regions.svg");
}
