require "octruby"

iris = Datamunge::DataFrame.iris
puts "iris: #{iris.nrows} rows x #{iris.ncols} cols\n\n"

puts "=== RBF kernel (default) ==="
rbf_model = Datamunge::SVM.new(iris, "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width")
rbf_model.print_summary
puts "\nConfusion matrix (rows = actual, cols = predicted):"
puts rbf_model.confusion_matrix.to_string

puts "\n=== Linear kernel, for comparison ==="
linear_model = Datamunge::SVM.new(iris, "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width", "linear")
puts "Training accuracy: #{linear_model.training_accuracy * 100.0}%"
puts "Support vectors: #{linear_model.num_support_vectors}"

newdata = Datamunge::DataFrame.new
newdata.add_numeric_column("Sepal.Length", [5.1, 6.0, 6.5, 6.2])
newdata.add_numeric_column("Sepal.Width", [3.5, 2.7, 3.0, 2.8])
newdata.add_numeric_column("Petal.Length", [1.4, 4.5, 5.5, 4.8])
newdata.add_numeric_column("Petal.Width", [0.2, 1.5, 2.0, 1.8])

puts "\nRBF predictions for new flowers (votes out of 3 one-vs-one pairs):"
puts rbf_model.predict_frame(newdata).to_string
