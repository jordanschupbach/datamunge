# CMD: just TARGET=forest_iris_ex run-julia

# Random forest classifier on iris: summary, confusion matrix, misclassifications, OOB accuracy.
using Datamunge

const iris = Datamunge.Data.iris()
println("iris: $(nrows(iris)) rows x $(ncols(iris)) cols\n")

const model = RandomForestClassifier(iris, "Species ~ Petal.Length + Petal.Width")
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

save(plot_classification(model, iris, "Petal.Length", "Petal.Width"), "forest_iris_classification.svg")
save(plot_decision_regions(model, "Petal.Length", "Petal.Width"), "forest_iris_decision_regions.svg")
println("\nSaved forest_iris_classification.svg and forest_iris_decision_regions.svg")

# A small forest, for comparison, showing how out-of-bag accuracy stabilizes as more trees
# are added -- the defining random forest effect a single decision tree cannot demonstrate.
# RandomForestClassifier(data, formula, n_trees, max_depth, min_samples_split, min_samples_leaf,
#                        max_features, criterion, bootstrap, sample_fraction, seed)
let small_forest = RandomForestClassifier(iris, "Species ~ Petal.Length + Petal.Width", 5)
    println("\n5-tree forest:   training accuracy=$(training_accuracy(small_forest) * 100)%  OOB accuracy=$(oob_accuracy(small_forest) * 100)%")
    println("100-tree forest: training accuracy=$(training_accuracy(model) * 100)%  OOB accuracy=$(oob_accuracy(model) * 100)%")
    save(plot_decision_regions(small_forest, "Petal.Length", "Petal.Width"), "forest_iris_decision_regions_5trees.svg")
    println("Saved forest_iris_decision_regions_5trees.svg")
end
