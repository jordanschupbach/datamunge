# See examples/r/lm_ex.r for notes on the flat ClassName_method(obj, ...) call convention.
# Note: DataFrame_numeric_at/string_at take a 0-based row index (matching the underlying
# C++ API); vectors returned to R (like predict()'s result) are indexed the normal R way (1-based).
library(datamunger)

iris <- DataFrame_iris()
n <- DataFrame_nrows(iris)
cat(n, "rows x", DataFrame_ncols(iris), "cols\n\n")

model <- DecisionTreeClassifier(iris, "Species ~ Petal.Length + Petal.Width")
DecisionTreeClassifier_print_summary(model)

cat("\nConfusion matrix (rows = actual, cols = predicted):\n")
cat(DataFrame_to_string(DecisionTreeClassifier_confusion_matrix(model)), "\n")

cat("\nMisclassified rows:\n")
predictions <- DecisionTreeClassifier_predict(model, iris)
misclassified <- 0
for (i in 0:(n - 1)) {
  actual <- DataFrame_string_at(iris, "Species", i)
  predicted <- predictions[i + 1]
  if (predicted == actual) next
  misclassified <- misclassified + 1
  cat("  row", i, ": Petal.Length=", DataFrame_numeric_at(iris, "Petal.Length", i),
      " Petal.Width=", DataFrame_numeric_at(iris, "Petal.Width", i),
      " actual=", actual, " predicted=", predicted, "\n")
}
cat(misclassified, "of", n, "misclassified (", 100.0 * misclassified / n, "%)\n")

Plot_save(DecisionTreeClassifier_plot_classification(model, iris, "Petal.Length", "Petal.Width"), "tree_iris_classification.svg")
Plot_save(DecisionTreeClassifier_plot_decision_regions(model, "Petal.Length", "Petal.Width"), "tree_iris_decision_regions.svg")
cat("\nSaved tree_iris_classification.svg and tree_iris_decision_regions.svg\n")

# A shallower tree, for comparison, showing a coarser (but still fairly accurate) decision
# boundary. DecisionTreeClassifier(data, formula, max_depth, min_samples_split, min_samples_leaf, criterion)
shallow <- DecisionTreeClassifier(iris, "Species ~ Petal.Length + Petal.Width", 2)
cat("\nDepth-2 tree training accuracy:", DecisionTreeClassifier_training_accuracy(shallow) * 100.0,
    "% (", DecisionTreeClassifier_leaf_count(shallow), "leaves)\n")
Plot_save(DecisionTreeClassifier_plot_decision_regions(shallow, "Petal.Length", "Petal.Width"), "tree_iris_decision_regions_depth2.svg")
cat("Saved tree_iris_decision_regions_depth2.svg\n")
