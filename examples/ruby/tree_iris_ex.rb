require "octruby"

iris = Datamunge::DataFrame.iris
puts "iris: #{iris.nrows} rows x #{iris.ncols} cols\n\n"

model = Datamunge::DecisionTreeClassifier.new(iris, "Species ~ Petal.Length + Petal.Width")
model.print_summary

puts "\nConfusion matrix (rows = actual, cols = predicted):"
puts model.confusion_matrix.to_string

puts "\nMisclassified rows:"
predictions = model.predict(iris)
misclassified = 0
iris.nrows.times do |i|
  actual = iris.string_at("Species", i)
  next if predictions[i] == actual
  misclassified += 1
  puts "  row #{i}: Petal.Length=#{iris.numeric_at('Petal.Length', i)} " \
       "Petal.Width=#{iris.numeric_at('Petal.Width', i)}  actual=#{actual}  predicted=#{predictions[i]}"
end
puts "#{misclassified} of #{iris.nrows} misclassified (#{100.0 * misclassified / iris.nrows}%)"

model.plot_classification(iris, "Petal.Length", "Petal.Width").save("tree_iris_classification.svg")
model.plot_decision_regions("Petal.Length", "Petal.Width").save("tree_iris_decision_regions.svg")
puts "\nSaved tree_iris_classification.svg and tree_iris_decision_regions.svg"

# A shallower tree, for comparison, showing a coarser (but still fairly accurate) decision boundary.
shallow = Datamunge::DecisionTreeClassifier.new(iris, "Species ~ Petal.Length + Petal.Width", 2)
puts "\nDepth-2 tree training accuracy: #{shallow.training_accuracy * 100.0}% (#{shallow.leaf_count} leaves)"
shallow.plot_decision_regions("Petal.Length", "Petal.Width").save("tree_iris_decision_regions_depth2.svg")
puts "Saved tree_iris_decision_regions_depth2.svg"
