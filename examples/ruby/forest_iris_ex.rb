require "octruby"

iris = Datamunge::DataFrame.iris
puts "iris: #{iris.nrows} rows x #{iris.ncols} cols\n\n"

model = Datamunge::RandomForestClassifier.new(iris, "Species ~ Petal.Length + Petal.Width")
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

model.plot_classification(iris, "Petal.Length", "Petal.Width").save("forest_iris_classification.svg")
model.plot_decision_regions("Petal.Length", "Petal.Width").save("forest_iris_decision_regions.svg")
puts "\nSaved forest_iris_classification.svg and forest_iris_decision_regions.svg"

# A small forest, for comparison, showing how out-of-bag accuracy stabilizes as more trees
# are added -- the defining random forest effect a single decision tree cannot demonstrate.
# RandomForestClassifier(data, formula, n_trees, max_depth, min_samples_split, min_samples_leaf,
#                         max_features, criterion, bootstrap, sample_fraction, seed)
small_forest = Datamunge::RandomForestClassifier.new(iris, "Species ~ Petal.Length + Petal.Width", 5)
puts "\n5-tree forest:   training accuracy=#{small_forest.training_accuracy * 100.0}%" \
     "  OOB accuracy=#{small_forest.oob_accuracy * 100.0}%"
puts "100-tree forest: training accuracy=#{model.training_accuracy * 100.0}%" \
     "  OOB accuracy=#{model.oob_accuracy * 100.0}%"
small_forest.plot_decision_regions("Petal.Length", "Petal.Width").save("forest_iris_decision_regions_5trees.svg")
puts "Saved forest_iris_decision_regions_5trees.svg"
