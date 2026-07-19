using System;

class Program {
  static void Main() {
    var penguins = DataFrame.penguins();
    Console.WriteLine($"penguins: {penguins.nrows()} rows x {penguins.ncols()} cols\n");

    var model = new NaiveBayesClassifier(penguins, "species ~ bill_length_mm + bill_depth_mm + island + sex");
    model.print_summary();

    Console.WriteLine("\nConfusion matrix (rows = actual, cols = predicted):");
    Console.WriteLine(model.confusion_matrix().to_string());

    Console.WriteLine("\nMisclassified rows:");
    var predictions = model.predict(penguins);
    int misclassified = 0;
    uint n = penguins.nrows();
    for (uint i = 0; i < n; i++) {
      if (!penguins.is_null("species", i) && !penguins.is_null("bill_length_mm", i) &&
          !penguins.is_null("bill_depth_mm", i) && !penguins.is_null("island", i) && !penguins.is_null("sex", i)) {
        var actual = penguins.string_at("species", i);
        var pred = predictions[(int)i];
        if (pred != actual) {
          misclassified += 1;
          Console.WriteLine($"  row {i}: bill_length={penguins.numeric_at("bill_length_mm", i)} bill_depth={penguins.numeric_at("bill_depth_mm", i)} island={penguins.string_at("island", i)} sex={penguins.string_at("sex", i)}  actual={actual}  predicted={pred}");
        }
      }
    }
    Console.WriteLine($"{misclassified} misclassified (of {n} rows, some incomplete)");

    var billOnly = new NaiveBayesClassifier(penguins, "species ~ bill_length_mm + bill_depth_mm");
    Console.WriteLine($"\nbill-measurements-only model training accuracy: {billOnly.training_accuracy() * 100.0}%");
    billOnly.plot_classification(penguins, "bill_length_mm", "bill_depth_mm").save("naive_bayes_penguins_classification.svg");
    billOnly.plot_decision_regions("bill_length_mm", "bill_depth_mm").save("naive_bayes_penguins_decision_regions.svg");
    Console.WriteLine("Saved naive_bayes_penguins_classification.svg and naive_bayes_penguins_decision_regions.svg");
  }
}
