# CMD: just TARGET=lda_iris_ex run-julia

# Linear discriminant analysis on iris: summary, confusion matrix, predictions and discriminant plot.
using Datamunge

const iris = Datamunge.Data.iris()
println("iris: $(nrows(iris)) rows x $(ncols(iris)) cols\n")

const model = LDA(iris, "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width")
print_summary(model)

println("\nConfusion matrix (rows = actual, cols = predicted):\n$(to_string(confusion_matrix(model)))")

let newdata = DataFrame()
    add_numeric_column(newdata, "Sepal.Length", [5.1, 6.0, 6.5, 6.2])
    add_numeric_column(newdata, "Sepal.Width", [3.5, 2.7, 3.0, 2.8])
    add_numeric_column(newdata, "Petal.Length", [1.4, 4.5, 5.5, 4.8])
    add_numeric_column(newdata, "Petal.Width", [0.2, 1.5, 2.0, 1.8])
    println("\nPredictions for new flowers:\n$(to_string(predict_frame(model, newdata)))")
end

save_discriminant_plot(model, "lda_iris_discriminants.svg")
println("\nSaved discriminant plot as lda_iris_discriminants.svg")
