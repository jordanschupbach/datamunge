# See examples/r/lm_ex.r for notes on the flat ClassName_method(obj, ...) call convention.
# Note: unlike the C++ example, this doesn't call drop_nulls() first (that method isn't
# exposed on the SWIG-bound DataFrame facade) -- NaiveBayesClassifier drops incomplete rows
# for its own formula columns internally when fitting, matching every other formula-based
# model in this library.
library(datamunger)

penguins <- DataFrame_penguins()
n <- DataFrame_nrows(penguins)
cat(n, "rows x", DataFrame_ncols(penguins), "cols\n\n")

model <- NaiveBayesClassifier(penguins, "species ~ bill_length_mm + bill_depth_mm + island + sex")
NaiveBayesClassifier_print_summary(model)

cat("\nConfusion matrix (rows = actual, cols = predicted):\n")
cat(DataFrame_to_string(NaiveBayesClassifier_confusion_matrix(model)), "\n")

cat("\nMisclassified rows:\n")
predictions <- NaiveBayesClassifier_predict(model, penguins)
misclassified <- 0
for (i in 0:(n - 1)) {
  if (DataFrame_is_null(penguins, "species", i) || DataFrame_is_null(penguins, "bill_length_mm", i) ||
      DataFrame_is_null(penguins, "bill_depth_mm", i) || DataFrame_is_null(penguins, "island", i) ||
      DataFrame_is_null(penguins, "sex", i)) next
  actual <- DataFrame_string_at(penguins, "species", i)
  predicted <- predictions[i + 1]
  if (predicted == actual) next
  misclassified <- misclassified + 1
  cat("  row", i, ": bill_length=", DataFrame_numeric_at(penguins, "bill_length_mm", i),
      " bill_depth=", DataFrame_numeric_at(penguins, "bill_depth_mm", i),
      " island=", DataFrame_string_at(penguins, "island", i), " sex=", DataFrame_string_at(penguins, "sex", i),
      " actual=", actual, " predicted=", predicted, "\n")
}
cat(misclassified, "misclassified (of", n, "rows, some incomplete)\n")

# plot_decision_regions requires exactly two NUMERIC predictors, so build a separate
# two-predictor model (bill measurements alone) just for visualization.
bill_only <- NaiveBayesClassifier(penguins, "species ~ bill_length_mm + bill_depth_mm")
cat("\nbill-measurements-only model training accuracy:", NaiveBayesClassifier_training_accuracy(bill_only) * 100.0, "%\n")
Plot_save(NaiveBayesClassifier_plot_classification(bill_only, penguins, "bill_length_mm", "bill_depth_mm"), "naive_bayes_penguins_classification.svg")
Plot_save(NaiveBayesClassifier_plot_decision_regions(bill_only, "bill_length_mm", "bill_depth_mm"), "naive_bayes_penguins_decision_regions.svg")
cat("Saved naive_bayes_penguins_classification.svg and naive_bayes_penguins_decision_regions.svg\n")
