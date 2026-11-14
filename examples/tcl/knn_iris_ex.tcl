# CMD: just TARGET=knn_iris_ex run-tcl
# k-nearest-neighbours classifier on iris: leave-one-out accuracy and decision regions for several k.
package require datamunge

set formula "Species ~ Petal.Length + Petal.Width"

set iris [datamunge::Data::iris]
puts "iris: [$iris nrows] rows x [$iris ncols] cols\n"

set model [datamunge::KNNClassifier new $iris $formula]
$model print_summary

puts "\nConfusion matrix (leave-one-out; rows = actual, cols = predicted):"
puts [[$model confusion_matrix] to_string]

puts "\nLeave-one-out misclassified rows:"
set fitted [$model predict $iris]
set n [$iris nrows]
set misclassified 0
for {set i 0} {$i < $n} {incr i} {
    set actual [$iris string_at Species $i]
    set predicted [lindex $fitted $i]
    if {$predicted eq $actual} {
        continue
    }
    incr misclassified
    puts "  row $i: Petal.Length=[$iris numeric_at Petal.Length $i]\
        Petal.Width=[$iris numeric_at Petal.Width $i]  actual=$actual  predicted=$predicted"
}
puts "$misclassified of $n misclassified ([expr {100.0 * $misclassified / $n}]%)"

[$model plot_decision_regions Petal.Length Petal.Width] save knn_iris_decision_regions_k5.svg
puts "\nSaved knn_iris_decision_regions_k5.svg"

# k=1 memorizes every training point exactly (jagged, overfit boundary with an island around
# every point, including noise); k=25 averages over a much larger neighborhood (very smooth,
# underfit boundary). KNNClassifier new data formula ?k? ?metric? ?weighted? ?standardize?
foreach k {1 25} {
    set knn [datamunge::KNNClassifier new $iris $formula $k]
    set file knn_iris_decision_regions_k$k.svg
    puts "\nk=[format %-2s $k] leave-one-out accuracy: [expr {[$knn training_accuracy] * 100}]%"
    [$knn plot_decision_regions Petal.Length Petal.Width] save $file
    puts "Saved $file"
}
