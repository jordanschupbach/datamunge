# CMD: nix develop .#r --command Rscript examples/r/xgboost_iris_ex.r
library(datamunger)

iris <- DataFrame$iris()
n <- iris$nrows()
cat(n, "rows x", iris$ncols(), "cols\n\n")

model <- XGBoostClassifier$new(iris, "Species ~ Petal.Length + Petal.Width")
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

model$plot_training_deviance()$save("xgboost_iris_training_deviance.svg")
model$plot_decision_regions("Petal.Length", "Petal.Width")$save("xgboost_iris_decision_regions.svg")
cat("\nSaved xgboost_iris_training_deviance.svg and xgboost_iris_decision_regions.svg\n")

# XGBoost's defining lever is regularization: a heavily L2-regularized model (large lambda)
# grows the same deep trees but keeps every leaf weight small, producing a much smoother
# decision boundary than the lightly-regularized default.
# XGBoostClassifier(data, formula, n_trees, learning_rate, max_depth, lambda, alpha, gamma,
#                    min_child_weight, min_samples_leaf, subsample, colsample_bytree, seed)
heavy <- XGBoostClassifier$new(iris, "Species ~ Petal.Length + Petal.Width", 100, 0.3, 6, 50.0)
model_dev <- model$training_deviance()
heavy_dev <- heavy$training_deviance()
cat("\nlambda=1 (default):   training accuracy=", model$training_accuracy() * 100.0,
    "%  deviance=", model_dev[length(model_dev)], "\n")
cat("lambda=50 (heavy L2): training accuracy=", heavy$training_accuracy() * 100.0,
    "%  deviance=", heavy_dev[length(heavy_dev)], "\n")
heavy$plot_decision_regions("Petal.Length", "Petal.Width")$save("xgboost_iris_decision_regions_heavy_lambda.svg")
cat("Saved xgboost_iris_decision_regions_heavy_lambda.svg\n")
