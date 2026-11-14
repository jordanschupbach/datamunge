# CMD: just TARGET=gbm_iris_ex run-ruby
# Gradient-boosted trees (GBM) on iris: summary, confusion matrix, misclassified rows, 5 vs 100 rounds.
require "datamunge"

iris = Datamunge::Data.iris
puts "iris: #{iris.nrows} rows x #{iris.ncols} cols\n\n"

model = Datamunge::GBMClassifier.new(iris, "Species ~ Petal.Length + Petal.Width")
model.print_summary

puts "\nConfusion matrix (rows = actual, cols = predicted):\n#{model.confusion_matrix}"

puts "\nMisclassified rows:"
predictions = model.predict(iris)
n = iris.nrows
misclassified = 0
n.times do |i|
  actual, predicted = iris.string_at("Species", i), predictions[i]
  next if predicted == actual
  misclassified += 1
  puts "  row #{i}: Petal.Length=#{iris.numeric_at("Petal.Length", i)} Petal.Width=#{iris.numeric_at("Petal.Width", i)}  actual=#{actual}  predicted=#{predicted}"
end
puts "#{misclassified} of #{n} misclassified (#{100.0 * misclassified / n}%)"

model.plot_training_deviance.save("gbm_iris_training_deviance.svg")
model.plot_decision_regions("Petal.Length", "Petal.Width").save("gbm_iris_decision_regions.svg")
puts "\nSaved gbm_iris_training_deviance.svg and gbm_iris_decision_regions.svg"

# A handful of boosting rounds vs a well-boosted ensemble: each additional round chips away
# at the training loss, gradually sharpening the decision boundary.
# GBMClassifier.new(data, formula, n_trees, learning_rate, max_depth, min_samples_split,
#                   min_samples_leaf, subsample, seed)
few = Datamunge::GBMClassifier.new(iris, "Species ~ Petal.Length + Petal.Width", 5)
puts "\n5-round ensemble:   training accuracy=#{100 * few.training_accuracy}%  deviance=#{few.training_deviance.last}"
puts "100-round ensemble: training accuracy=#{100 * model.training_accuracy}%  deviance=#{model.training_deviance.last}"
few.plot_decision_regions("Petal.Length", "Petal.Width").save("gbm_iris_decision_regions_5rounds.svg")
puts "Saved gbm_iris_decision_regions_5rounds.svg"
