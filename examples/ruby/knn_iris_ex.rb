require "octruby"

iris = Datamunge::DataFrame.iris
puts "iris: #{iris.nrows} rows x #{iris.ncols} cols\n\n"

model = Datamunge::KNNClassifier.new(iris, "Species ~ Petal.Length + Petal.Width")
model.print_summary

puts "\nConfusion matrix (leave-one-out; rows = actual, cols = predicted):"
puts model.confusion_matrix.to_string

puts "\nLeave-one-out misclassified rows:"
fitted = model.predict(iris)
misclassified = 0
iris.nrows.times do |i|
  actual = iris.string_at("Species", i)
  next if fitted[i] == actual
  misclassified += 1
  puts "  row #{i}: Petal.Length=#{iris.numeric_at('Petal.Length', i)} " \
       "Petal.Width=#{iris.numeric_at('Petal.Width', i)}  actual=#{actual}  predicted=#{fitted[i]}"
end
puts "#{misclassified} of #{iris.nrows} misclassified (#{100.0 * misclassified / iris.nrows}%)"

model.plot_decision_regions("Petal.Length", "Petal.Width").save("knn_iris_decision_regions_k5.svg")
puts "\nSaved knn_iris_decision_regions_k5.svg"

# k=1 memorizes every training point exactly (jagged, overfit boundary with an island around
# every point, including noise); k=25 averages over a much larger neighborhood (very smooth,
# underfit boundary). KNNClassifier(data, formula, k, metric, weighted, standardize)
k1 = Datamunge::KNNClassifier.new(iris, "Species ~ Petal.Length + Petal.Width", 1)
puts "\nk=1  leave-one-out accuracy: #{k1.training_accuracy * 100.0}%"
k1.plot_decision_regions("Petal.Length", "Petal.Width").save("knn_iris_decision_regions_k1.svg")
puts "Saved knn_iris_decision_regions_k1.svg"

k25 = Datamunge::KNNClassifier.new(iris, "Species ~ Petal.Length + Petal.Width", 25)
puts "\nk=25 leave-one-out accuracy: #{k25.training_accuracy * 100.0}%"
k25.plot_decision_regions("Petal.Length", "Petal.Width").save("knn_iris_decision_regions_k25.svg")
puts "Saved knn_iris_decision_regions_k25.svg"
