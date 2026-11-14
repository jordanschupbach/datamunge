# CMD: just TARGET=xgboost_iris_ex run-tcl
# XGBoost classifier on iris: summary, confusion matrix, misclassified rows, L2-regularization comparison.
package require datamunge

set iris [datamunge::Data::iris]
puts "iris: [$iris nrows] rows x [$iris ncols] cols\n"

set model [datamunge::XGBoostClassifier new $iris "Species ~ Petal.Length + Petal.Width"]
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

[$model plot_training_deviance] save xgboost_iris_training_deviance.svg
[$model plot_decision_regions Petal.Length Petal.Width] save xgboost_iris_decision_regions.svg
puts "\nSaved xgboost_iris_training_deviance.svg and xgboost_iris_decision_regions.svg"

# XGBoost's defining lever is regularization: a heavily L2-regularized model (large lambda)
# grows the same deep trees but keeps every leaf weight small, producing a much smoother
# decision boundary than the lightly-regularized default.
# datamunge::XGBoostClassifier new data formula ?n_trees? ?learning_rate? ?max_depth? ?lambda?
#     ?alpha? ?gamma? ?min_child_weight? ?min_samples_leaf? ?subsample? ?colsample_bytree? ?seed?
set heavy [datamunge::XGBoostClassifier new $iris "Species ~ Petal.Length + Petal.Width" 100 0.3 6 50.0]
puts "\nlambda=1 (default):   training accuracy=[expr {100 * [$model training_accuracy]}]%  deviance=[lindex [$model training_deviance] end]"
puts "lambda=50 (heavy L2): training accuracy=[expr {100 * [$heavy training_accuracy]}]%  deviance=[lindex [$heavy training_deviance] end]"
[$heavy plot_decision_regions Petal.Length Petal.Width] save xgboost_iris_decision_regions_heavy_lambda.svg
puts "Saved xgboost_iris_decision_regions_heavy_lambda.svg"
