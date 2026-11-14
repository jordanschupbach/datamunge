# CMD: just TARGET=knn_iris_ex run-ruby
# k-nearest-neighbours classifier on iris: leave-one-out accuracy and decision regions for several k.
require "datamunge"

iris = Datamunge::Data.iris
puts "iris: #{iris.nrows} rows x #{iris.ncols} cols\n\n"

model = Datamunge::KNNClassifier.new(iris, "Species ~ Petal.Length + Petal.Width")
model.print_summary

puts "\nConfusion matrix (leave-one-out; rows = actual, cols = predicted):"
puts model.confusion_matrix

puts "\nLeave-one-out misclassified rows:"
fitted = model.predict(iris)
n = iris.nrows
misclassified = 0
n.times do |i|
  actual, predicted = iris.string_at("Species", i), fitted[i]
  next if predicted == actual
  misclassified += 1
  puts "  row #{i}: Petal.Length=#{iris.numeric_at("Petal.Length", i)} Petal.Width=#{iris.numeric_at("Petal.Width", i)}  actual=#{actual}  predicted=#{predicted}"
end
puts "#{misclassified} of #{n} misclassified (#{100.0 * misclassified / n}%)"

model.plot_decision_regions("Petal.Length", "Petal.Width").save("knn_iris_decision_regions_k5.svg")
puts "\nSaved knn_iris_decision_regions_k5.svg"

# k=1 memorizes every training point exactly (jagged, overfit boundary with an island around
# every point, including noise); k=25 averages over a much larger neighborhood (very smooth,
# underfit boundary). KNNClassifier.new(data, formula, k, metric, weighted, standardize)
[1, 25].each do |k|
  knn = Datamunge::KNNClassifier.new(iris, "Species ~ Petal.Length + Petal.Width", k)
  file = "knn_iris_decision_regions_k#{k}.svg"
  puts "\nk=#{k.to_s.ljust(2)} leave-one-out accuracy: #{knn.training_accuracy * 100}%"
  knn.plot_decision_regions("Petal.Length", "Petal.Width").save(file)
  puts "Saved #{file}"
end
