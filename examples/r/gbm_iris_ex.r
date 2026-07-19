# See examples/r/lm_ex.r for notes on the flat ClassName_method(obj, ...) call convention.
library(datamunger)

iris <- DataFrame_iris()
n <- DataFrame_nrows(iris)
cat(n, "rows x", DataFrame_ncols(iris), "cols\n\n")

model <- GBMClassifier(iris, "Species ~ Petal.Length + Petal.Width")
GBMClassifier_print_summary(model)

cat("\nConfusion matrix (rows = actual, cols = predicted):\n")
cat(DataFrame_to_string(GBMClassifier_confusion_matrix(model)), "\n")

cat("\nMisclassified rows:\n")
predictions <- GBMClassifier_predict(model, iris)
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

Plot_save(GBMClassifier_plot_training_deviance(model), "gbm_iris_training_deviance.svg")
Plot_save(GBMClassifier_plot_decision_regions(model, "Petal.Length", "Petal.Width"), "gbm_iris_decision_regions.svg")
cat("\nSaved gbm_iris_training_deviance.svg and gbm_iris_decision_regions.svg\n")

# A handful of boosting rounds vs a well-boosted ensemble: each additional round chips away
# at the training loss, gradually sharpening the decision boundary.
# GBMClassifier(data, formula, n_trees, learning_rate, max_depth, min_samples_split, min_samples_leaf, subsample, seed)
few <- GBMClassifier(iris, "Species ~ Petal.Length + Petal.Width", 5)
few_dev <- GBMClassifier_training_deviance(few)
model_dev <- GBMClassifier_training_deviance(model)
cat("\n5-round ensemble:   training accuracy=", GBMClassifier_training_accuracy(few) * 100.0,
    "%  deviance=", few_dev[length(few_dev)], "\n")
cat("100-round ensemble: training accuracy=", GBMClassifier_training_accuracy(model) * 100.0,
    "%  deviance=", model_dev[length(model_dev)], "\n")
Plot_save(GBMClassifier_plot_decision_regions(few, "Petal.Length", "Petal.Width"), "gbm_iris_decision_regions_5rounds.svg")
cat("Saved gbm_iris_decision_regions_5rounds.svg\n")
