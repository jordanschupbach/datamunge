# See examples/r/lm_ex.r for notes on the flat ClassName_method(obj, ...) call convention.
library(datamunger)

iris <- DataFrame_iris()
n <- DataFrame_nrows(iris)
cat(n, "rows x", DataFrame_ncols(iris), "cols\n\n")

model <- KNNClassifier(iris, "Species ~ Petal.Length + Petal.Width")
KNNClassifier_print_summary(model)

cat("\nConfusion matrix (leave-one-out; rows = actual, cols = predicted):\n")
cat(DataFrame_to_string(KNNClassifier_confusion_matrix(model)), "\n")

cat("\nLeave-one-out misclassified rows:\n")
fitted <- KNNClassifier_predict(model, iris)
misclassified <- 0
for (i in 0:(n - 1)) {
  actual <- DataFrame_string_at(iris, "Species", i)
  predicted <- fitted[i + 1]
  if (predicted == actual) next
  misclassified <- misclassified + 1
  cat("  row", i, ": Petal.Length=", DataFrame_numeric_at(iris, "Petal.Length", i),
      " Petal.Width=", DataFrame_numeric_at(iris, "Petal.Width", i),
      " actual=", actual, " predicted=", predicted, "\n")
}
cat(misclassified, "of", n, "misclassified (", 100.0 * misclassified / n, "%)\n")

Plot_save(KNNClassifier_plot_decision_regions(model, "Petal.Length", "Petal.Width"), "knn_iris_decision_regions_k5.svg")
cat("\nSaved knn_iris_decision_regions_k5.svg\n")

# k=1 memorizes every training point exactly (jagged, overfit boundary); k=25 averages over a
# much larger neighborhood (very smooth, underfit boundary).
# KNNClassifier(data, formula, k, metric, weighted, standardize)
k1 <- KNNClassifier(iris, "Species ~ Petal.Length + Petal.Width", 1)
cat("\nk=1  leave-one-out accuracy:", KNNClassifier_training_accuracy(k1) * 100.0, "%\n")
Plot_save(KNNClassifier_plot_decision_regions(k1, "Petal.Length", "Petal.Width"), "knn_iris_decision_regions_k1.svg")
cat("Saved knn_iris_decision_regions_k1.svg\n")

k25 <- KNNClassifier(iris, "Species ~ Petal.Length + Petal.Width", 25)
cat("\nk=25 leave-one-out accuracy:", KNNClassifier_training_accuracy(k25) * 100.0, "%\n")
Plot_save(KNNClassifier_plot_decision_regions(k25, "Petal.Length", "Petal.Width"), "knn_iris_decision_regions_k25.svg")
cat("Saved knn_iris_decision_regions_k25.svg\n")
