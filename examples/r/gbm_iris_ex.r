# CMD: nix develop .#r --command Rscript examples/r/gbm_iris_ex.r
library(datamunger)

iris <- DataFrame$iris()
n <- iris$nrows()
cat(n, "rows x", iris$ncols(), "cols\n\n")

model <- GBMClassifier$new(iris, "Species ~ Petal.Length + Petal.Width")
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

model$plot_training_deviance()$save("gbm_iris_training_deviance.svg")
model$plot_decision_regions("Petal.Length", "Petal.Width")$save("gbm_iris_decision_regions.svg")
cat("\nSaved gbm_iris_training_deviance.svg and gbm_iris_decision_regions.svg\n")

# A handful of boosting rounds vs a well-boosted ensemble: each additional round chips away
# at the training loss, gradually sharpening the decision boundary.
# GBMClassifier(data, formula, n_trees, learning_rate, max_depth, min_samples_split, min_samples_leaf, subsample, seed)
few <- GBMClassifier$new(iris, "Species ~ Petal.Length + Petal.Width", 5)
few_dev <- few$training_deviance()
model_dev <- model$training_deviance()
cat("\n5-round ensemble:   training accuracy=", few$training_accuracy() * 100.0,
    "%  deviance=", few_dev[length(few_dev)], "\n")
cat("100-round ensemble: training accuracy=", model$training_accuracy() * 100.0,
    "%  deviance=", model_dev[length(model_dev)], "\n")
few$plot_decision_regions("Petal.Length", "Petal.Width")$save("gbm_iris_decision_regions_5rounds.svg")
cat("Saved gbm_iris_decision_regions_5rounds.svg\n")
