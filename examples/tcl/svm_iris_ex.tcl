package require Datamunge 0.0.1

set iris [datamunge::DataFrame_iris]
puts "iris: [datamunge::DataFrame_nrows $iris] rows x [datamunge::DataFrame_ncols $iris] cols\n"

puts "=== RBF kernel (default) ==="
set rbf_model [datamunge::new_SVM $iris "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width"]
datamunge::SVM_print_summary $rbf_model
puts "\nConfusion matrix (rows = actual, cols = predicted):"
puts "[datamunge::DataFrame_to_string [datamunge::SVM_confusion_matrix $rbf_model]]"

puts "\n=== Linear kernel, for comparison ==="
set linear_model [datamunge::new_SVM $iris "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width" "linear"]
puts "Training accuracy: [expr {[datamunge::SVM_training_accuracy $linear_model] * 100.0}]%"
puts "Support vectors: [datamunge::SVM_num_support_vectors $linear_model]"

set newdata [datamunge::DataFrame_empty]
datamunge::DataFrame_add_numeric_column $newdata "Sepal.Length" {5.1 6.0 6.5 6.2}
datamunge::DataFrame_add_numeric_column $newdata "Sepal.Width" {3.5 2.7 3.0 2.8}
datamunge::DataFrame_add_numeric_column $newdata "Petal.Length" {1.4 4.5 5.5 4.8}
datamunge::DataFrame_add_numeric_column $newdata "Petal.Width" {0.2 1.5 2.0 1.8}

puts "\nRBF predictions for new flowers (votes out of 3 one-vs-one pairs):"
puts "[datamunge::DataFrame_to_string [datamunge::SVM_predict_frame $rbf_model $newdata]]"
