# CMD: just TARGET=tree_iris_ex run-tcl
# Decision tree classifier on iris: summary, confusion matrix, misclassified rows and plots.
package require datamunge

set iris [datamunge::Data::iris]
puts "iris: [$iris nrows] rows x [$iris ncols] cols\n"

set model [datamunge::DecisionTreeClassifier new $iris "Species ~ Petal.Length + Petal.Width"]
$model print_summary

puts "\nConfusion matrix (rows = actual, cols = predicted):"
puts [[$model confusion_matrix] to_string]

puts "\nMisclassified rows:"
set predictions [$model predict $iris]
set n [$iris nrows]
set misclassified 0
set i -1
foreach predicted $predictions {
    incr i
    set actual [$iris string_at Species $i]
    if {$predicted eq $actual} {
        continue
    }
    incr misclassified
    puts "  row $i: Petal.Length=[$iris numeric_at Petal.Length $i] Petal.Width=[$iris numeric_at Petal.Width $i]  actual=$actual  predicted=$predicted"
}
puts "$misclassified of $n misclassified ([expr {100.0 * $misclassified / $n}]%)"

[$model plot_classification $iris Petal.Length Petal.Width] save tree_iris_classification.svg
[$model plot_decision_regions Petal.Length Petal.Width] save tree_iris_decision_regions.svg
puts "\nSaved tree_iris_classification.svg and tree_iris_decision_regions.svg"

# A shallower tree, for comparison, showing a coarser (but still fairly accurate) decision boundary.
set shallow [datamunge::DecisionTreeClassifier new $iris "Species ~ Petal.Length + Petal.Width" 2]
puts "\nDepth-2 tree training accuracy: [expr {[$shallow training_accuracy] * 100}]% ([$shallow leaf_count] leaves)"
[$shallow plot_decision_regions Petal.Length Petal.Width] save tree_iris_decision_regions_depth2.svg
puts "Saved tree_iris_decision_regions_depth2.svg"
