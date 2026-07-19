# See examples/r/lm_ex.r for notes on the flat ClassName_method(obj, ...) call convention.
library(datamunger)

iris <- DataFrame_iris()
cat(DataFrame_nrows(iris), "rows x", DataFrame_ncols(iris), "cols\n\n")

cat("=== RBF kernel (default) ===\n")
rbf_model <- SVM(iris, "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width")
SVM_print_summary(rbf_model)
cat("\nConfusion matrix (rows = actual, cols = predicted):\n")
cat(DataFrame_to_string(SVM_confusion_matrix(rbf_model)), "\n")

cat("\n=== Linear kernel, for comparison ===\n")
linear_model <- SVM(iris, "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width", "linear")
cat("Training accuracy:", SVM_training_accuracy(linear_model) * 100.0, "%\n")
cat("Support vectors:", SVM_num_support_vectors(linear_model), "\n")

newdata <- DataFrame_empty()
DataFrame_add_numeric_column(newdata, "Sepal.Length", c(5.1, 6.0, 6.5, 6.2))
DataFrame_add_numeric_column(newdata, "Sepal.Width", c(3.5, 2.7, 3.0, 2.8))
DataFrame_add_numeric_column(newdata, "Petal.Length", c(1.4, 4.5, 5.5, 4.8))
DataFrame_add_numeric_column(newdata, "Petal.Width", c(0.2, 1.5, 2.0, 1.8))

cat("\nRBF predictions for new flowers (votes out of 3 one-vs-one pairs):\n")
cat(DataFrame_to_string(SVM_predict_frame(rbf_model, newdata)), "\n")
