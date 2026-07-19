# See examples/r/lm_ex.r for notes on the flat ClassName_method(obj, ...) call convention.
library(datamunger)

iris <- DataFrame_iris()
cat(DataFrame_nrows(iris), "rows x", DataFrame_ncols(iris), "cols\n\n")

model <- LDA(iris, "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width")
LDA_print_summary(model)

cat("\nConfusion matrix (rows = actual, cols = predicted):\n")
cat(DataFrame_to_string(LDA_confusion_matrix(model)), "\n")

newdata <- DataFrame_empty()
DataFrame_add_numeric_column(newdata, "Sepal.Length", c(5.1, 6.0, 6.5, 6.2))
DataFrame_add_numeric_column(newdata, "Sepal.Width", c(3.5, 2.7, 3.0, 2.8))
DataFrame_add_numeric_column(newdata, "Petal.Length", c(1.4, 4.5, 5.5, 4.8))
DataFrame_add_numeric_column(newdata, "Petal.Width", c(0.2, 1.5, 2.0, 1.8))

cat("\nPredictions for new flowers:\n")
cat(DataFrame_to_string(LDA_predict_frame(model, newdata)), "\n")

LDA_save_discriminant_plot(model, "lda_iris_discriminants.svg")
cat("\nSaved discriminant plot as lda_iris_discriminants.svg\n")
