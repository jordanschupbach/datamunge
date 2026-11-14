# CMD: just TARGET=tree_iris_ex run-julia

# Decision tree classifier on iris: summary, confusion matrix, misclassified rows and plots.
using Datamunge

const iris = Datamunge.Data.iris()
println("iris: $(nrows(iris)) rows x $(ncols(iris)) cols\n")

const model = DecisionTreeClassifier(iris, "Species ~ Petal.Length + Petal.Width")
print_summary(model)

println("\nConfusion matrix (rows = actual, cols = predicted):")
println(to_string(confusion_matrix(model)))

println("\nMisclassified rows:")
let predictions = predict(model, iris), n = nrows(iris), misclassified = 0
    for i in 0:n-1
        actual, predicted = string_at(iris, "Species", i), predictions[i + 1]
        predicted == actual && continue
        misclassified += 1
        println("  row $i: Petal.Length=$(numeric_at(iris, "Petal.Length", i)) Petal.Width=$(numeric_at(iris, "Petal.Width", i))  actual=$actual  predicted=$predicted")
    end
    println("$misclassified of $n misclassified ($(100 * misclassified / n)%)")
end

save(plot_classification(model, iris, "Petal.Length", "Petal.Width"), "tree_iris_classification.svg")
save(plot_decision_regions(model, "Petal.Length", "Petal.Width"), "tree_iris_decision_regions.svg")
println("\nSaved tree_iris_classification.svg and tree_iris_decision_regions.svg")

# A shallower tree, for comparison, showing a coarser (but still fairly accurate) decision boundary.
let shallow = DecisionTreeClassifier(iris, "Species ~ Petal.Length + Petal.Width", 2)
    println("\nDepth-2 tree training accuracy: $(training_accuracy(shallow) * 100)% ($(leaf_count(shallow)) leaves)")
    save(plot_decision_regions(shallow, "Petal.Length", "Petal.Width"), "tree_iris_decision_regions_depth2.svg")
    println("Saved tree_iris_decision_regions_depth2.svg")
end
