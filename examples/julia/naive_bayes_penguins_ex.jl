# CMD: just TARGET=naive_bayes_penguins_ex run-julia

# Naive Bayes on penguins: mixed numeric/categorical predictors, confusion matrix, misclassified rows, plots.
using Datamunge

# Note: NaiveBayesClassifier drops incomplete rows for its own formula columns internally
# when fitting, matching every other formula-based model in this library.
const penguins = Datamunge.Data.penguins()
println("penguins: $(nrows(penguins)) rows x $(ncols(penguins)) cols\n")

# A mix of numeric (Gaussian likelihood) and categorical (frequency-table likelihood)
# predictors in one formula -- each modeled independently given the class, per the naive
# Bayes assumption.
const model = NaiveBayesClassifier(penguins, "species ~ bill_length_mm + bill_depth_mm + island + sex")
print_summary(model)

println("\nConfusion matrix (rows = actual, cols = predicted):\n$(to_string(confusion_matrix(model)))")

println("\nMisclassified rows:")
let predictions = predict(model, penguins), misclassified = 0
    columns = ("species", "bill_length_mm", "bill_depth_mm", "island", "sex")
    for i in 0:nrows(penguins)-1
        any(c -> is_null(penguins, c, i), columns) && continue
        actual, predicted = string_at(penguins, "species", i), predictions[i + 1]
        predicted == actual && continue
        misclassified += 1
        println("  row $i: bill_length=$(numeric_at(penguins, "bill_length_mm", i)) bill_depth=$(numeric_at(penguins, "bill_depth_mm", i)) " *
                "island=$(string_at(penguins, "island", i)) sex=$(string_at(penguins, "sex", i))  actual=$actual  predicted=$predicted")
    end
    println("$misclassified misclassified (of $(nrows(penguins)) rows, some incomplete)")
end

# plot_decision_regions requires exactly two NUMERIC predictors, so build a separate
# two-predictor model (bill measurements alone) just for visualization.
let bill_only = NaiveBayesClassifier(penguins, "species ~ bill_length_mm + bill_depth_mm")
    println("\nbill-measurements-only model training accuracy: $(100 * training_accuracy(bill_only))%")
    save(plot_classification(bill_only, penguins, "bill_length_mm", "bill_depth_mm"), "naive_bayes_penguins_classification.svg")
    save(plot_decision_regions(bill_only, "bill_length_mm", "bill_depth_mm"), "naive_bayes_penguins_decision_regions.svg")
    println("Saved naive_bayes_penguins_classification.svg and naive_bayes_penguins_decision_regions.svg")
end
