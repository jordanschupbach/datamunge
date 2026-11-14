# CMD: just TARGET=xgboost_iris_ex run-julia

# XGBoost classifier on iris: summary, confusion matrix, misclassified rows, L2-regularization comparison.
using Datamunge

const iris = Datamunge.Data.iris()
println("iris: $(nrows(iris)) rows x $(ncols(iris)) cols\n")

const model = XGBoostClassifier(iris, "Species ~ Petal.Length + Petal.Width")
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

save(plot_training_deviance(model), "xgboost_iris_training_deviance.svg")
save(plot_decision_regions(model, "Petal.Length", "Petal.Width"), "xgboost_iris_decision_regions.svg")
println("\nSaved xgboost_iris_training_deviance.svg and xgboost_iris_decision_regions.svg")

# XGBoost's defining lever is regularization: a heavily L2-regularized model (large lambda)
# grows the same deep trees but keeps every leaf weight small, producing a much smoother
# decision boundary than the lightly-regularized default.
# XGBoostClassifier(data, formula, n_trees, learning_rate, max_depth, lambda, alpha, gamma,
#                   min_child_weight, min_samples_leaf, subsample, colsample_bytree, seed)
let heavy = XGBoostClassifier(iris, "Species ~ Petal.Length + Petal.Width", 100, 0.3, 6, 50.0)
    println("\nlambda=1 (default):   training accuracy=$(100 * training_accuracy(model))%  deviance=$(last(training_deviance(model)))")
    println("lambda=50 (heavy L2): training accuracy=$(100 * training_accuracy(heavy))%  deviance=$(last(training_deviance(heavy)))")
    save(plot_decision_regions(heavy, "Petal.Length", "Petal.Width"), "xgboost_iris_decision_regions_heavy_lambda.svg")
    println("Saved xgboost_iris_decision_regions_heavy_lambda.svg")
end
