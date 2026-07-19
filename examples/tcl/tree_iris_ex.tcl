package require Datamunge 0.0.1

set iris [datamunge::DataFrame_iris]
puts "iris: [datamunge::DataFrame_nrows $iris] rows x [datamunge::DataFrame_ncols $iris] cols\n"

set model [datamunge::new_DecisionTreeClassifier $iris "Species ~ Petal.Length + Petal.Width"]
datamunge::DecisionTreeClassifier_print_summary $model

puts "\nConfusion matrix (rows = actual, cols = predicted):"
puts "[datamunge::DataFrame_to_string [datamunge::DecisionTreeClassifier_confusion_matrix $model]]"

puts "\nMisclassified rows:"
set predictions [datamunge::DecisionTreeClassifier_predict $model $iris]
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

datamunge::Plot_save [datamunge::DecisionTreeClassifier_plot_classification $model $iris "Petal.Length" "Petal.Width"] "tree_iris_classification.svg"
datamunge::Plot_save [datamunge::DecisionTreeClassifier_plot_decision_regions $model "Petal.Length" "Petal.Width"] "tree_iris_decision_regions.svg"
puts "\nSaved tree_iris_classification.svg and tree_iris_decision_regions.svg"

set shallow [datamunge::new_DecisionTreeClassifier $iris "Species ~ Petal.Length + Petal.Width" 2]
puts "\nDepth-2 tree training accuracy: [expr {[datamunge::DecisionTreeClassifier_training_accuracy $shallow] * 100.0}]% ([datamunge::DecisionTreeClassifier_leaf_count $shallow] leaves)"
datamunge::Plot_save [datamunge::DecisionTreeClassifier_plot_decision_regions $shallow "Petal.Length" "Petal.Width"] "tree_iris_decision_regions_depth2.svg"
puts "Saved tree_iris_decision_regions_depth2.svg"
