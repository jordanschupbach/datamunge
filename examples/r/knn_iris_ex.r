# CMD: nix develop .#r --command env R_LIBS_USER=build/r/library Rscript examples/r/knn_iris_ex.r
# REPL: nix develop .#r --command env R_LIBS_USER=build/r/library R --no-save --quiet
library(datamunger)

iris <- data$iris()
n <- iris$nrows()
cat(n, "rows x", iris$ncols(), "cols\n\n")

model <- KNNClassifier$new(iris, "Species ~ Petal.Length + Petal.Width")
model$print_summary()

cat("\nConfusion matrix (leave-one-out; rows = actual, cols = predicted):\n")
cat(model$confusion_matrix()$to_string(), "\n")

cat("\nLeave-one-out misclassified rows:\n")
fitted <- model$predict(iris)
misclassified <- 0
for (i in 0:(n - 1)) {
  actual <- iris$string_at("Species", i)
  predicted <- fitted[i + 1]
  if (predicted == actual) next
  misclassified <- misclassified + 1
  cat("  row", i, ": Petal.Length=", iris$numeric_at("Petal.Length", i),
      " Petal.Width=", iris$numeric_at("Petal.Width", i),
      " actual=", actual, " predicted=", predicted, "\n")
}
cat(misclassified, "of", n, "misclassified (", 100.0 * misclassified / n, "%)\n")

model$plot_decision_regions("Petal.Length", "Petal.Width")$save("knn_iris_decision_regions_k5.svg")
cat("\nSaved knn_iris_decision_regions_k5.svg\n")

# k=1 memorizes every training point exactly (jagged, overfit boundary); k=25 averages over a
# much larger neighborhood (very smooth, underfit boundary).
# KNNClassifier(data, formula, k, metric, weighted, standardize)
k1 <- KNNClassifier$new(iris, "Species ~ Petal.Length + Petal.Width", 1)
cat("\nk=1  leave-one-out accuracy:", k1$training_accuracy() * 100.0, "%\n")
k1$plot_decision_regions("Petal.Length", "Petal.Width")$save("knn_iris_decision_regions_k1.svg")
cat("Saved knn_iris_decision_regions_k1.svg\n")

k25 <- KNNClassifier$new(iris, "Species ~ Petal.Length + Petal.Width", 25)
cat("\nk=25 leave-one-out accuracy:", k25$training_accuracy() * 100.0, "%\n")
k25$plot_decision_regions("Petal.Length", "Petal.Width")$save("knn_iris_decision_regions_k25.svg")
cat("Saved knn_iris_decision_regions_k25.svg\n")
