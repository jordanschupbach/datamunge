# See examples/r/lm_ex.r for notes on the flat ClassName_method(obj, ...) call convention.
library(datamunger)

iris <- DataFrame_iris()
n <- DataFrame_nrows(iris)
cat(n, "rows x", DataFrame_ncols(iris), "cols\n\n")

model <- XGBoostClassifier(iris, "Species ~ Petal.Length + Petal.Width")
XGBoostClassifier_print_summary(model)

cat("\nConfusion matrix (rows = actual, cols = predicted):\n")
cat(DataFrame_to_string(XGBoostClassifier_confusion_matrix(model)), "\n")

cat("\nMisclassified rows:\n")
predictions <- XGBoostClassifier_predict(model, iris)
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

Plot_save(XGBoostClassifier_plot_training_deviance(model), "xgboost_iris_training_deviance.svg")
Plot_save(XGBoostClassifier_plot_decision_regions(model, "Petal.Length", "Petal.Width"), "xgboost_iris_decision_regions.svg")
cat("\nSaved xgboost_iris_training_deviance.svg and xgboost_iris_decision_regions.svg\n")

# XGBoost's defining lever is regularization: a heavily L2-regularized model (large lambda)
# grows the same deep trees but keeps every leaf weight small, producing a much smoother
# decision boundary than the lightly-regularized default.
# XGBoostClassifier(data, formula, n_trees, learning_rate, max_depth, lambda, alpha, gamma,
#                    min_child_weight, min_samples_leaf, subsample, colsample_bytree, seed)
heavy <- XGBoostClassifier(iris, "Species ~ Petal.Length + Petal.Width", 100, 0.3, 6, 50.0)
model_dev <- XGBoostClassifier_training_deviance(model)
heavy_dev <- XGBoostClassifier_training_deviance(heavy)
cat("\nlambda=1 (default):   training accuracy=", XGBoostClassifier_training_accuracy(model) * 100.0,
    "%  deviance=", model_dev[length(model_dev)], "\n")
cat("lambda=50 (heavy L2): training accuracy=", XGBoostClassifier_training_accuracy(heavy) * 100.0,
    "%  deviance=", heavy_dev[length(heavy_dev)], "\n")
Plot_save(XGBoostClassifier_plot_decision_regions(heavy, "Petal.Length", "Petal.Width"), "xgboost_iris_decision_regions_heavy_lambda.svg")
cat("Saved xgboost_iris_decision_regions_heavy_lambda.svg\n")
