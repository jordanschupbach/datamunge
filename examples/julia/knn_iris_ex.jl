# CMD: just TARGET=knn_iris_ex run-julia

# k-nearest-neighbours classifier on iris: leave-one-out accuracy and decision regions for several k.
using Datamunge

const iris = Datamunge.Data.iris()
println("iris: $(nrows(iris)) rows x $(ncols(iris)) cols\n")

const model = KNNClassifier(iris, "Species ~ Petal.Length + Petal.Width")
print_summary(model)

println("\nConfusion matrix (leave-one-out; rows = actual, cols = predicted):")
println(to_string(confusion_matrix(model)))

println("\nLeave-one-out misclassified rows:")
let fitted = predict(model, iris), n = nrows(iris), misclassified = 0
    for i in 0:n-1
        actual, predicted = string_at(iris, "Species", i), fitted[i + 1]
        predicted == actual && continue
        misclassified += 1
        println("  row $i: Petal.Length=$(numeric_at(iris, "Petal.Length", i)) Petal.Width=$(numeric_at(iris, "Petal.Width", i))  actual=$actual  predicted=$predicted")
    end
    println("$misclassified of $n misclassified ($(100 * misclassified / n)%)")
end

save(plot_decision_regions(model, "Petal.Length", "Petal.Width"), "knn_iris_decision_regions_k5.svg")
println("\nSaved knn_iris_decision_regions_k5.svg")

# k=1 memorizes every training point exactly (jagged, overfit boundary with an island around
# every point, including noise); k=25 averages over a much larger neighborhood (very smooth,
# underfit boundary). KNNClassifier(data, formula, k, metric, weighted, standardize)
for k in (1, 25)
    knn = KNNClassifier(iris, "Species ~ Petal.Length + Petal.Width", k)
    file = "knn_iris_decision_regions_k$k.svg"
    println("\nk=$(rpad(k, 2)) leave-one-out accuracy: $(training_accuracy(knn) * 100)%")
    save(plot_decision_regions(knn, "Petal.Length", "Petal.Width"), file)
    println("Saved $file")
end
