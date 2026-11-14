# CMD: just TARGET=gbm_iris_ex run-julia

# Gradient-boosted trees (GBM) on iris: summary, confusion matrix, misclassified rows, 5 vs 100 rounds.
using Datamunge

const iris = Datamunge.Data.iris()
println("iris: $(nrows(iris)) rows x $(ncols(iris)) cols\n")

const model = GBMClassifier(iris, "Species ~ Petal.Length + Petal.Width")
print_summary(model)

println("\nConfusion matrix (rows = actual, cols = predicted):\n$(to_string(confusion_matrix(model)))")

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

save(plot_training_deviance(model), "gbm_iris_training_deviance.svg")
save(plot_decision_regions(model, "Petal.Length", "Petal.Width"), "gbm_iris_decision_regions.svg")
println("\nSaved gbm_iris_training_deviance.svg and gbm_iris_decision_regions.svg")

# A handful of boosting rounds vs a well-boosted ensemble: each additional round chips away
# at the training loss, gradually sharpening the decision boundary.
# GBMClassifier(data, formula, n_trees, learning_rate, max_depth, min_samples_split,
#               min_samples_leaf, subsample, seed)
let few = GBMClassifier(iris, "Species ~ Petal.Length + Petal.Width", 5)
    println("\n5-round ensemble:   training accuracy=$(100 * training_accuracy(few))%  deviance=$(last(training_deviance(few)))")
    println("100-round ensemble: training accuracy=$(100 * training_accuracy(model))%  deviance=$(last(training_deviance(model)))")
    save(plot_decision_regions(few, "Petal.Length", "Petal.Width"), "gbm_iris_decision_regions_5rounds.svg")
    println("Saved gbm_iris_decision_regions_5rounds.svg")
end
