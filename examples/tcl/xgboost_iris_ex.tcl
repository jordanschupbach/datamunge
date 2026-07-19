package require Datamunge 0.0.1

set iris [datamunge::DataFrame_iris]
puts "iris: [datamunge::DataFrame_nrows $iris] rows x [datamunge::DataFrame_ncols $iris] cols\n"

set model [datamunge::new_XGBoostClassifier $iris "Species ~ Petal.Length + Petal.Width"]
datamunge::XGBoostClassifier_print_summary $model

puts "\nConfusion matrix (rows = actual, cols = predicted):"
puts "[datamunge::DataFrame_to_string [datamunge::XGBoostClassifier_confusion_matrix $model]]"

puts "\nMisclassified rows:"
set predictions [datamunge::XGBoostClassifier_predict $model $iris]
set misclassified 0
set n [datamunge::DataFrame_nrows $iris]
for {set i 0} {$i < $n} {incr i} {
  set actual [datamunge::DataFrame_string_at $iris "Species" $i]
  set pred [lindex $predictions $i]
  if {$pred ne $actual} {
    incr misclassified
    set pl [datamunge::DataFrame_numeric_at $iris "Petal.Length" $i]
    set pw [datamunge::DataFrame_numeric_at $iris "Petal.Width" $i]
    puts "  row $i: Petal.Length=$pl Petal.Width=$pw  actual=$actual  predicted=$pred"
  }
}
puts "$misclassified of $n misclassified ([expr {100.0 * $misclassified / $n}]%)"

datamunge::Plot_save [datamunge::XGBoostClassifier_plot_training_deviance $model] "xgboost_iris_training_deviance.svg"
datamunge::Plot_save [datamunge::XGBoostClassifier_plot_decision_regions $model "Petal.Length" "Petal.Width"] "xgboost_iris_decision_regions.svg"
puts "\nSaved xgboost_iris_training_deviance.svg and xgboost_iris_decision_regions.svg"

set heavy [datamunge::new_XGBoostClassifier $iris "Species ~ Petal.Length + Petal.Width" 100 0.3 6 50.0]
set model_dev [datamunge::XGBoostClassifier_training_deviance $model]
set heavy_dev [datamunge::XGBoostClassifier_training_deviance $heavy]
puts "\nlambda=1 (default):   training accuracy=[expr {[datamunge::XGBoostClassifier_training_accuracy $model] * 100.0}]%  deviance=[lindex $model_dev end]"
puts "lambda=50 (heavy L2): training accuracy=[expr {[datamunge::XGBoostClassifier_training_accuracy $heavy] * 100.0}]%  deviance=[lindex $heavy_dev end]"
datamunge::Plot_save [datamunge::XGBoostClassifier_plot_decision_regions $heavy "Petal.Length" "Petal.Width"] "xgboost_iris_decision_regions_heavy_lambda.svg"
puts "Saved xgboost_iris_decision_regions_heavy_lambda.svg"
