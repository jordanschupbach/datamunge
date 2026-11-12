# CMD: nix develop .#r --command Rscript examples/r/naive_bayes_penguins_ex.r
# REPL: nix develop .#r --command R --no-save --quiet
# Note: unlike the C++ example, this doesn't call drop_nulls() first (that method isn't
# exposed on the bound DataFrame facade) -- NaiveBayesClassifier drops incomplete rows
# for its own formula columns internally when fitting, matching every other formula-based
# model in this library.
library(datamunger)

penguins <- DataFrame$penguins()
n <- penguins$nrows()
cat(n, "rows x", penguins$ncols(), "cols\n\n")

model <- NaiveBayesClassifier$new(penguins, "species ~ bill_length_mm + bill_depth_mm + island + sex")
model$print_summary()

cat("\nConfusion matrix (rows = actual, cols = predicted):\n")
cat(model$confusion_matrix()$to_string(), "\n")

cat("\nMisclassified rows:\n")
predictions <- model$predict(penguins)
misclassified <- 0
for (i in 0:(n - 1)) {
  if (penguins$is_null("species", i) || penguins$is_null("bill_length_mm", i) ||
      penguins$is_null("bill_depth_mm", i) || penguins$is_null("island", i) ||
      penguins$is_null("sex", i)) next
  actual <- penguins$string_at("species", i)
  predicted <- predictions[i + 1]
  if (predicted == actual) next
  misclassified <- misclassified + 1
  cat("  row", i, ": bill_length=", penguins$numeric_at("bill_length_mm", i),
      " bill_depth=", penguins$numeric_at("bill_depth_mm", i),
      " island=", penguins$string_at("island", i), " sex=", penguins$string_at("sex", i),
      " actual=", actual, " predicted=", predicted, "\n")
}
cat(misclassified, "misclassified (of", n, "rows, some incomplete)\n")

# plot_decision_regions requires exactly two NUMERIC predictors, so build a separate
# two-predictor model (bill measurements alone) just for visualization.
bill_only <- NaiveBayesClassifier$new(penguins, "species ~ bill_length_mm + bill_depth_mm")
cat("\nbill-measurements-only model training accuracy:", bill_only$training_accuracy() * 100.0, "%\n")
bill_only$plot_classification(penguins, "bill_length_mm", "bill_depth_mm")$save("naive_bayes_penguins_classification.svg")
bill_only$plot_decision_regions("bill_length_mm", "bill_depth_mm")$save("naive_bayes_penguins_decision_regions.svg")
cat("Saved naive_bayes_penguins_classification.svg and naive_bayes_penguins_decision_regions.svg\n")
