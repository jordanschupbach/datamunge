# See examples/r/lm_ex.r for notes on the flat ClassName_method(obj, ...) call convention.
library(datamunger)

iris <- DataFrame_iris()
n <- DataFrame_nrows(iris)
cat(n, "rows x", DataFrame_ncols(iris), "cols\n\n")

model <- RandomForestClassifier(iris, "Species ~ Petal.Length + Petal.Width")
RandomForestClassifier_print_summary(model)

cat("\nConfusion matrix (rows = actual, cols = predicted):\n")
cat(DataFrame_to_string(RandomForestClassifier_confusion_matrix(model)), "\n")

cat("\nMisclassified rows:\n")
predictions <- RandomForestClassifier_predict(model, iris)
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

Plot_save(RandomForestClassifier_plot_classification(model, iris, "Petal.Length", "Petal.Width"), "forest_iris_classification.svg")
Plot_save(RandomForestClassifier_plot_decision_regions(model, "Petal.Length", "Petal.Width"), "forest_iris_decision_regions.svg")
cat("\nSaved forest_iris_classification.svg and forest_iris_decision_regions.svg\n")

# A small forest, for comparison, showing how out-of-bag accuracy stabilizes as more trees
# are added. RandomForestClassifier(data, formula, n_trees, max_depth, min_samples_split,
#                                    min_samples_leaf, max_features, criterion, bootstrap, sample_fraction, seed)
small_forest <- RandomForestClassifier(iris, "Species ~ Petal.Length + Petal.Width", 5)
cat("\n5-tree forest:   training accuracy=", RandomForestClassifier_training_accuracy(small_forest) * 100.0,
    "%  OOB accuracy=", RandomForestClassifier_oob_accuracy(small_forest) * 100.0, "%\n")
cat("100-tree forest: training accuracy=", RandomForestClassifier_training_accuracy(model) * 100.0,
    "%  OOB accuracy=", RandomForestClassifier_oob_accuracy(model) * 100.0, "%\n")
Plot_save(RandomForestClassifier_plot_decision_regions(small_forest, "Petal.Length", "Petal.Width"), "forest_iris_decision_regions_5trees.svg")
cat("Saved forest_iris_decision_regions_5trees.svg\n")
