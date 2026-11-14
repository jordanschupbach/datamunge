# CMD: just TARGET=svm_iris_ex run-julia

# Support vector machine on iris: RBF vs. linear kernels, confusion matrix and predictions.
using Datamunge

const formula = "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width"
const iris = Datamunge.Data.iris()
println("iris: $(nrows(iris)) rows x $(ncols(iris)) cols\n")

println("=== RBF kernel (default) ===")
const rbf_model = SVM(iris, formula)
print_summary(rbf_model)
println("\nConfusion matrix (rows = actual, cols = predicted):\n$(to_string(confusion_matrix(rbf_model)))")

println("\n=== Linear kernel, for comparison ===")
let linear_model = SVM(iris, formula, "linear")
    println("Training accuracy: $(training_accuracy(linear_model) * 100)%")
    println("Support vectors: $(num_support_vectors(linear_model))")
end

let newdata = DataFrame()
    add_numeric_column(newdata, "Sepal.Length", [5.1, 6.0, 6.5, 6.2])
    add_numeric_column(newdata, "Sepal.Width", [3.5, 2.7, 3.0, 2.8])
    add_numeric_column(newdata, "Petal.Length", [1.4, 4.5, 5.5, 4.8])
    add_numeric_column(newdata, "Petal.Width", [0.2, 1.5, 2.0, 1.8])
    println("\nRBF predictions for new flowers (votes out of 3 one-vs-one pairs):\n$(to_string(predict_frame(rbf_model, newdata)))")
end
