1;

datamunge;

% Note: unlike the C++ example, this doesn't call drop_nulls() first (that method isn't
% exposed on the SWIG-bound DataFrame facade) -- NaiveBayesClassifier drops incomplete rows
% for its own formula columns internally when fitting, matching every other formula-based
% model in this library.
penguins = DataFrame_penguins();
printf("penguins: %d rows x %d cols\n\n", DataFrame_nrows(penguins), DataFrame_ncols(penguins));

% A mix of numeric (Gaussian likelihood) and categorical (frequency-table likelihood)
% predictors in one formula -- each modeled independently given the class, per the naive
% Bayes assumption.
model = NaiveBayesClassifier(penguins, "species ~ bill_length_mm + bill_depth_mm + island + sex");
NaiveBayesClassifier_print_summary(model);

printf("\nConfusion matrix (rows = actual, cols = predicted):\n");
printf("%s\n", DataFrame_to_string(NaiveBayesClassifier_confusion_matrix(model)));

printf("\nMisclassified rows:\n");
predictions = NaiveBayesClassifier_predict(model, penguins);
misclassified = 0;
n = DataFrame_nrows(penguins);
for i = 0:(n - 1)
  if !(DataFrame_is_null(penguins, "species", i) || DataFrame_is_null(penguins, "bill_length_mm", i) || ...
       DataFrame_is_null(penguins, "bill_depth_mm", i) || DataFrame_is_null(penguins, "island", i) || DataFrame_is_null(penguins, "sex", i))
    actual = DataFrame_string_at(penguins, "species", i);
    pred = predictions{i + 1};
    if !strcmp(pred, actual)
      misclassified = misclassified + 1;
      printf("  row %d: bill_length=%g bill_depth=%g island=%s sex=%s  actual=%s  predicted=%s\n", i, ...
             DataFrame_numeric_at(penguins, "bill_length_mm", i), DataFrame_numeric_at(penguins, "bill_depth_mm", i), ...
             DataFrame_string_at(penguins, "island", i), DataFrame_string_at(penguins, "sex", i), actual, pred);
    end
  end
end
printf("%d misclassified (of %d rows, some incomplete)\n", misclassified, n);

% plot_decision_regions requires exactly two NUMERIC predictors, so build a separate
% two-predictor model (bill measurements alone) just for visualization.
bill_only = NaiveBayesClassifier(penguins, "species ~ bill_length_mm + bill_depth_mm");
printf("\nbill-measurements-only model training accuracy: %g%%\n", NaiveBayesClassifier_training_accuracy(bill_only) * 100.0);
Plot_save(NaiveBayesClassifier_plot_classification(bill_only, penguins, "bill_length_mm", "bill_depth_mm"), "naive_bayes_penguins_classification.svg");
Plot_save(NaiveBayesClassifier_plot_decision_regions(bill_only, "bill_length_mm", "bill_depth_mm"), "naive_bayes_penguins_decision_regions.svg");
printf("Saved naive_bayes_penguins_classification.svg and naive_bayes_penguins_decision_regions.svg\n");
