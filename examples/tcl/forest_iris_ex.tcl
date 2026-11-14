# CMD: just TARGET=forest_iris_ex run-tcl
# Random forest classifier on iris: summary, confusion matrix, misclassifications, OOB accuracy.
package require datamunge

set iris [datamunge::Data::iris]
puts "iris: [$iris nrows] rows x [$iris ncols] cols\n"

set model [datamunge::RandomForestClassifier new $iris "Species ~ Petal.Length + Petal.Width"]
$model print_summary

puts "\nConfusion matrix (rows = actual, cols = predicted):"
puts [[$model confusion_matrix] to_string]

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

[$model plot_classification $iris Petal.Length Petal.Width] save forest_iris_classification.svg
[$model plot_decision_regions Petal.Length Petal.Width] save forest_iris_decision_regions.svg
puts "\nSaved forest_iris_classification.svg and forest_iris_decision_regions.svg"

# A small forest, for comparison, showing how out-of-bag accuracy stabilizes as more trees
# are added -- the defining random forest effect a single decision tree cannot demonstrate.
# datamunge::RandomForestClassifier new data formula ?n_trees? ?max_depth? ?min_samples_split?
#     ?min_samples_leaf? ?max_features? ?criterion? ?bootstrap? ?sample_fraction? ?seed?
set small_forest [datamunge::RandomForestClassifier new $iris "Species ~ Petal.Length + Petal.Width" 5]
puts "\n5-tree forest:   training accuracy=[expr {[$small_forest training_accuracy] * 100}]%  OOB accuracy=[expr {[$small_forest oob_accuracy] * 100}]%"
puts "100-tree forest: training accuracy=[expr {[$model training_accuracy] * 100}]%  OOB accuracy=[expr {[$model oob_accuracy] * 100}]%"
[$small_forest plot_decision_regions Petal.Length Petal.Width] save forest_iris_decision_regions_5trees.svg
puts "Saved forest_iris_decision_regions_5trees.svg"
