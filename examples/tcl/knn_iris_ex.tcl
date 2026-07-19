package require Datamunge 0.0.1

set iris [datamunge::DataFrame_iris]
puts "iris: [datamunge::DataFrame_nrows $iris] rows x [datamunge::DataFrame_ncols $iris] cols\n"

set model [datamunge::new_KNNClassifier $iris "Species ~ Petal.Length + Petal.Width"]
datamunge::KNNClassifier_print_summary $model

puts "\nConfusion matrix (leave-one-out; rows = actual, cols = predicted):"
puts "[datamunge::DataFrame_to_string [datamunge::KNNClassifier_confusion_matrix $model]]"

puts "\nLeave-one-out misclassified rows:"
set fitted [datamunge::KNNClassifier_predict $model $iris]
set misclassified 0
set n [datamunge::DataFrame_nrows $iris]
for {set i 0} {$i < $n} {incr i} {
  set actual [datamunge::DataFrame_string_at $iris "Species" $i]
  set pred [lindex $fitted $i]
  if {$pred ne $actual} {
    incr misclassified
    set pl [datamunge::DataFrame_numeric_at $iris "Petal.Length" $i]
    set pw [datamunge::DataFrame_numeric_at $iris "Petal.Width" $i]
    puts "  row $i: Petal.Length=$pl Petal.Width=$pw  actual=$actual  predicted=$pred"
  }
}
puts "$misclassified of $n misclassified ([expr {100.0 * $misclassified / $n}]%)"

datamunge::Plot_save [datamunge::KNNClassifier_plot_decision_regions $model "Petal.Length" "Petal.Width"] "knn_iris_decision_regions_k5.svg"
puts "\nSaved knn_iris_decision_regions_k5.svg"

set k1 [datamunge::new_KNNClassifier $iris "Species ~ Petal.Length + Petal.Width" 1]
puts "\nk=1  leave-one-out accuracy: [expr {[datamunge::KNNClassifier_training_accuracy $k1] * 100.0}]%"
datamunge::Plot_save [datamunge::KNNClassifier_plot_decision_regions $k1 "Petal.Length" "Petal.Width"] "knn_iris_decision_regions_k1.svg"
puts "Saved knn_iris_decision_regions_k1.svg"

set k25 [datamunge::new_KNNClassifier $iris "Species ~ Petal.Length + Petal.Width" 25]
puts "\nk=25 leave-one-out accuracy: [expr {[datamunge::KNNClassifier_training_accuracy $k25] * 100.0}]%"
datamunge::Plot_save [datamunge::KNNClassifier_plot_decision_regions $k25 "Petal.Length" "Petal.Width"] "knn_iris_decision_regions_k25.svg"
puts "Saved knn_iris_decision_regions_k25.svg"
