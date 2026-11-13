# CMD: nix develop .#r --command env R_LIBS_USER=build/r/library Rscript examples/r/tree_iris_ex.r
# REPL: nix develop .#r --command env R_LIBS_USER=build/r/library R --no-save --quiet
# Note: DataFrame_numeric_at/string_at take a 0-based row index (matching the underlying
# C++ API); vectors returned to R (like predict()'s result) are indexed the normal R way (1-based).
library(datamunger)

iris <- data$iris()
n <- iris$nrows()
cat(n, "rows x", iris$ncols(), "cols\n\n")

model <- DecisionTreeClassifier$new(iris, "Species ~ Petal.Length + Petal.Width")
model$print_summary()

cat("\nConfusion matrix (rows = actual, cols = predicted):\n")
cat(model$confusion_matrix()$to_string(), "\n")

cat("\nMisclassified rows:\n")
predictions <- model$predict(iris)
misclassified <- 0
for (i in 0:(n - 1)) {
  actual <- iris$string_at("Species", i)
  predicted <- predictions[i + 1]
  if (predicted == actual) next
  misclassified <- misclassified + 1
  cat("  row", i, ": Petal.Length=", iris$numeric_at("Petal.Length", i),
      " Petal.Width=", iris$numeric_at("Petal.Width", i),
      " actual=", actual, " predicted=", predicted, "\n")
}
cat(misclassified, "of", n, "misclassified (", 100.0 * misclassified / n, "%)\n")

model$plot_classification(iris, "Petal.Length", "Petal.Width")$save("tree_iris_classification.svg")
model$plot_decision_regions("Petal.Length", "Petal.Width")$save("tree_iris_decision_regions.svg")
cat("\nSaved tree_iris_classification.svg and tree_iris_decision_regions.svg\n")

# A shallower tree, for comparison, showing a coarser (but still fairly accurate) decision
# boundary. DecisionTreeClassifier(data, formula, max_depth, min_samples_split, min_samples_leaf, criterion)
shallow <- DecisionTreeClassifier$new(iris, "Species ~ Petal.Length + Petal.Width", 2)
cat("\nDepth-2 tree training accuracy:", shallow$training_accuracy() * 100.0,
    "% (", shallow$leaf_count(), "leaves)\n")
shallow$plot_decision_regions("Petal.Length", "Petal.Width")$save("tree_iris_decision_regions_depth2.svg")
cat("Saved tree_iris_decision_regions_depth2.svg\n")
