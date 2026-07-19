require "octruby"

iris = Datamunge::DataFrame.iris
puts "iris: #{iris.nrows} rows x #{iris.ncols} cols\n\n"

model = Datamunge::GBMClassifier.new(iris, "Species ~ Petal.Length + Petal.Width")
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

model.plot_training_deviance.save("gbm_iris_training_deviance.svg")
model.plot_decision_regions("Petal.Length", "Petal.Width").save("gbm_iris_decision_regions.svg")
puts "\nSaved gbm_iris_training_deviance.svg and gbm_iris_decision_regions.svg"

# A handful of boosting rounds vs a well-boosted ensemble: each additional round chips away
# at the training loss, gradually sharpening the decision boundary.
# GBMClassifier(data, formula, n_trees, learning_rate, max_depth, min_samples_split, min_samples_leaf, subsample, seed)
few = Datamunge::GBMClassifier.new(iris, "Species ~ Petal.Length + Petal.Width", 5)
puts "\n5-round ensemble:   training accuracy=#{few.training_accuracy * 100.0}%" \
     "  deviance=#{few.training_deviance[-1]}"
puts "100-round ensemble: training accuracy=#{model.training_accuracy * 100.0}%" \
     "  deviance=#{model.training_deviance[-1]}"
few.plot_decision_regions("Petal.Length", "Petal.Width").save("gbm_iris_decision_regions_5rounds.svg")
puts "Saved gbm_iris_decision_regions_5rounds.svg"
