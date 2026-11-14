# CMD: just TARGET=svm_iris_ex run-tcl
# Support vector machine on iris: RBF vs. linear kernels, confusion matrix and predictions.
package require datamunge

set formula "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width"
set iris [datamunge::Data::iris]
puts "iris: [$iris nrows] rows x [$iris ncols] cols\n"

puts "=== RBF kernel (default) ==="
set rbf_model [datamunge::SVM new $iris $formula]
$rbf_model print_summary
puts "\nConfusion matrix (rows = actual, cols = predicted):\n[[$rbf_model confusion_matrix] to_string]"

puts "\n=== Linear kernel, for comparison ==="
set linear_model [datamunge::SVM new $iris $formula linear]
puts "Training accuracy: [expr {[$linear_model training_accuracy] * 100}]%"
puts "Support vectors: [$linear_model num_support_vectors]"

set newdata [datamunge::DataFrame new]
$newdata add_numeric_column Sepal.Length {5.1 6.0 6.5 6.2}
$newdata add_numeric_column Sepal.Width {3.5 2.7 3.0 2.8}
$newdata add_numeric_column Petal.Length {1.4 4.5 5.5 4.8}
$newdata add_numeric_column Petal.Width {0.2 1.5 2.0 1.8}
puts "\nRBF predictions for new flowers (votes out of 3 one-vs-one pairs):\n[[$rbf_model predict_frame $newdata] to_string]"
