# CMD: nix develop .#r --command Rscript examples/r/svm_iris_ex.r
# REPL: nix develop .#r --command R --no-save --quiet
library(datamunger)

iris <- DataFrame$iris()
cat(iris$nrows(), "rows x", iris$ncols(), "cols\n\n")

cat("=== RBF kernel (default) ===\n")
rbf_model <- SVM$new(iris, "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width")
rbf_model$print_summary()
cat("\nConfusion matrix (rows = actual, cols = predicted):\n")
cat(rbf_model$confusion_matrix()$to_string(), "\n")

cat("\n=== Linear kernel, for comparison ===\n")
linear_model <- SVM$new(iris, "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width", "linear")
cat("Training accuracy:", linear_model$training_accuracy() * 100.0, "%\n")
cat("Support vectors:", linear_model$num_support_vectors(), "\n")

newdata <- DataFrame$empty()
newdata$add_numeric_column("Sepal.Length", c(5.1, 6.0, 6.5, 6.2))
newdata$add_numeric_column("Sepal.Width", c(3.5, 2.7, 3.0, 2.8))
newdata$add_numeric_column("Petal.Length", c(1.4, 4.5, 5.5, 4.8))
newdata$add_numeric_column("Petal.Width", c(0.2, 1.5, 2.0, 1.8))

cat("\nRBF predictions for new flowers (votes out of 3 one-vs-one pairs):\n")
cat(rbf_model$predict_frame(newdata)$to_string(), "\n")
