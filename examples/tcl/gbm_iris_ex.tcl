package require Datamunge 0.0.1

set iris [datamunge::DataFrame_iris]
puts "iris: [datamunge::DataFrame_nrows $iris] rows x [datamunge::DataFrame_ncols $iris] cols\n"

set model [datamunge::new_GBMClassifier $iris "Species ~ Petal.Length + Petal.Width"]
datamunge::GBMClassifier_print_summary $model

puts "\nConfusion matrix (rows = actual, cols = predicted):"
puts "[datamunge::DataFrame_to_string [datamunge::GBMClassifier_confusion_matrix $model]]"

puts "\nMisclassified rows:"
set predictions [datamunge::GBMClassifier_predict $model $iris]
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

datamunge::Plot_save [datamunge::GBMClassifier_plot_training_deviance $model] "gbm_iris_training_deviance.svg"
datamunge::Plot_save [datamunge::GBMClassifier_plot_decision_regions $model "Petal.Length" "Petal.Width"] "gbm_iris_decision_regions.svg"
puts "\nSaved gbm_iris_training_deviance.svg and gbm_iris_decision_regions.svg"

set few [datamunge::new_GBMClassifier $iris "Species ~ Petal.Length + Petal.Width" 5]
set few_dev [datamunge::GBMClassifier_training_deviance $few]
set model_dev [datamunge::GBMClassifier_training_deviance $model]
puts "\n5-round ensemble:   training accuracy=[expr {[datamunge::GBMClassifier_training_accuracy $few] * 100.0}]%  deviance=[lindex $few_dev end]"
puts "100-round ensemble: training accuracy=[expr {[datamunge::GBMClassifier_training_accuracy $model] * 100.0}]%  deviance=[lindex $model_dev end]"
datamunge::Plot_save [datamunge::GBMClassifier_plot_decision_regions $few "Petal.Length" "Petal.Width"] "gbm_iris_decision_regions_5rounds.svg"
puts "Saved gbm_iris_decision_regions_5rounds.svg"
