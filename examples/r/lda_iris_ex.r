# CMD: nix develop .#r --command Rscript examples/r/lda_iris_ex.r
# REPL: nix develop .#r --command R --no-save --quiet
library(datamunger)

iris <- DataFrame$iris()
cat(iris$nrows(), "rows x", iris$ncols(), "cols\n\n")

model <- LDA$new(iris, "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width")
model$print_summary()

cat("\nConfusion matrix (rows = actual, cols = predicted):\n")
cat(model$confusion_matrix()$to_string(), "\n")

newdata <- DataFrame$empty()
newdata$add_numeric_column("Sepal.Length", c(5.1, 6.0, 6.5, 6.2))
newdata$add_numeric_column("Sepal.Width", c(3.5, 2.7, 3.0, 2.8))
newdata$add_numeric_column("Petal.Length", c(1.4, 4.5, 5.5, 4.8))
newdata$add_numeric_column("Petal.Width", c(0.2, 1.5, 2.0, 1.8))

cat("\nPredictions for new flowers:\n")
cat(model$predict_frame(newdata)$to_string(), "\n")

model$save_discriminant_plot("lda_iris_discriminants.svg")
cat("\nSaved discriminant plot as lda_iris_discriminants.svg\n")
