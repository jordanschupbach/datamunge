# CMD: nix develop .#r --command Rscript examples/r/forest_iris_ex.r
library(datamunger)

iris <- DataFrame$iris()
n <- iris$nrows()
cat(n, "rows x", iris$ncols(), "cols\n\n")

model <- RandomForestClassifier$new(iris, "Species ~ Petal.Length + Petal.Width")
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

model$plot_classification(iris, "Petal.Length", "Petal.Width")$save("forest_iris_classification.svg")
model$plot_decision_regions("Petal.Length", "Petal.Width")$save("forest_iris_decision_regions.svg")
cat("\nSaved forest_iris_classification.svg and forest_iris_decision_regions.svg\n")

# A small forest, for comparison, showing how out-of-bag accuracy stabilizes as more trees
# are added. RandomForestClassifier(data, formula, n_trees, max_depth, min_samples_split,
#                                    min_samples_leaf, max_features, criterion, bootstrap, sample_fraction, seed)
small_forest <- RandomForestClassifier$new(iris, "Species ~ Petal.Length + Petal.Width", 5)
cat("\n5-tree forest:   training accuracy=", small_forest$training_accuracy() * 100.0,
    "%  OOB accuracy=", small_forest$oob_accuracy() * 100.0, "%\n")
cat("100-tree forest: training accuracy=", model$training_accuracy() * 100.0,
    "%  OOB accuracy=", model$oob_accuracy() * 100.0, "%\n")
small_forest$plot_decision_regions("Petal.Length", "Petal.Width")$save("forest_iris_decision_regions_5trees.svg")
cat("Saved forest_iris_decision_regions_5trees.svg\n")
