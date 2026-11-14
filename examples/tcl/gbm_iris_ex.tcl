# CMD: just TARGET=gbm_iris_ex run-tcl
# Gradient-boosted trees (GBM) on iris: summary, confusion matrix, misclassified rows, 5 vs 100 rounds.
package require datamunge

set iris [datamunge::Data::iris]
puts "iris: [$iris nrows] rows x [$iris ncols] cols\n"

set model [datamunge::GBMClassifier new $iris "Species ~ Petal.Length + Petal.Width"]
$model print_summary

puts "\nConfusion matrix (rows = actual, cols = predicted):\n[[$model confusion_matrix] to_string]"

puts "\nMisclassified rows:"
set predictions [$model predict $iris]
set n [$iris nrows]
set misclassified 0
for {set i 0} {$i < $n} {incr i} {
    set actual [$iris string_at Species $i]
    set predicted [lindex $predictions $i]
    if {$predicted eq $actual} {
        continue
    }
    incr misclassified
    puts "  row $i: Petal.Length=[$iris numeric_at Petal.Length $i] Petal.Width=[$iris numeric_at Petal.Width $i]  actual=$actual  predicted=$predicted"
}
puts "$misclassified of $n misclassified ([expr {100.0 * $misclassified / $n}]%)"

[$model plot_training_deviance] save gbm_iris_training_deviance.svg
[$model plot_decision_regions Petal.Length Petal.Width] save gbm_iris_decision_regions.svg
puts "\nSaved gbm_iris_training_deviance.svg and gbm_iris_decision_regions.svg"

# A handful of boosting rounds vs a well-boosted ensemble: each additional round chips away
# at the training loss, gradually sharpening the decision boundary.
# datamunge::GBMClassifier new data formula ?n_trees? ?learning_rate? ?max_depth?
#     ?min_samples_split? ?min_samples_leaf? ?subsample? ?seed?
set few [datamunge::GBMClassifier new $iris "Species ~ Petal.Length + Petal.Width" 5]
puts "\n5-round ensemble:   training accuracy=[expr {100 * [$few training_accuracy]}]%  deviance=[lindex [$few training_deviance] end]"
puts "100-round ensemble: training accuracy=[expr {100 * [$model training_accuracy]}]%  deviance=[lindex [$model training_deviance] end]"
[$few plot_decision_regions Petal.Length Petal.Width] save gbm_iris_decision_regions_5rounds.svg
puts "Saved gbm_iris_decision_regions_5rounds.svg"
