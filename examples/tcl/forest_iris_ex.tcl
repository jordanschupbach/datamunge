package require Datamunge 0.0.1

set iris [datamunge::DataFrame_iris]
puts "iris: [datamunge::DataFrame_nrows $iris] rows x [datamunge::DataFrame_ncols $iris] cols\n"

set model [datamunge::new_RandomForestClassifier $iris "Species ~ Petal.Length + Petal.Width"]
datamunge::RandomForestClassifier_print_summary $model

puts "\nConfusion matrix (rows = actual, cols = predicted):"
puts "[datamunge::DataFrame_to_string [datamunge::RandomForestClassifier_confusion_matrix $model]]"

puts "\nMisclassified rows:"
set predictions [datamunge::RandomForestClassifier_predict $model $iris]
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

datamunge::Plot_save [datamunge::RandomForestClassifier_plot_classification $model $iris "Petal.Length" "Petal.Width"] "forest_iris_classification.svg"
datamunge::Plot_save [datamunge::RandomForestClassifier_plot_decision_regions $model "Petal.Length" "Petal.Width"] "forest_iris_decision_regions.svg"
puts "\nSaved forest_iris_classification.svg and forest_iris_decision_regions.svg"

set small_forest [datamunge::new_RandomForestClassifier $iris "Species ~ Petal.Length + Petal.Width" 5]
puts "\n5-tree forest:   training accuracy=[expr {[datamunge::RandomForestClassifier_training_accuracy $small_forest] * 100.0}]%  OOB accuracy=[expr {[datamunge::RandomForestClassifier_oob_accuracy $small_forest] * 100.0}]%"
puts "100-tree forest: training accuracy=[expr {[datamunge::RandomForestClassifier_training_accuracy $model] * 100.0}]%  OOB accuracy=[expr {[datamunge::RandomForestClassifier_oob_accuracy $model] * 100.0}]%"
datamunge::Plot_save [datamunge::RandomForestClassifier_plot_decision_regions $small_forest "Petal.Length" "Petal.Width"] "forest_iris_decision_regions_5trees.svg"
puts "Saved forest_iris_decision_regions_5trees.svg"
