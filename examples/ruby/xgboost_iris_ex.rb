# CMD: just TARGET=xgboost_iris_ex run-ruby
# XGBoost classifier on iris: summary, confusion matrix, misclassified rows, L2-regularization comparison.
require "datamunge"

iris = Datamunge::Data.iris
puts "iris: #{iris.nrows} rows x #{iris.ncols} cols\n\n"

model = Datamunge::XGBoostClassifier.new(iris, "Species ~ Petal.Length + Petal.Width")
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

model.plot_training_deviance.save("xgboost_iris_training_deviance.svg")
model.plot_decision_regions("Petal.Length", "Petal.Width").save("xgboost_iris_decision_regions.svg")
puts "\nSaved xgboost_iris_training_deviance.svg and xgboost_iris_decision_regions.svg"

# XGBoost's defining lever is regularization: a heavily L2-regularized model (large lambda)
# grows the same deep trees but keeps every leaf weight small, producing a much smoother
# decision boundary than the lightly-regularized default.
# XGBoostClassifier.new(data, formula, n_trees, learning_rate, max_depth, lambda, alpha, gamma,
#                       min_child_weight, min_samples_leaf, subsample, colsample_bytree, seed)
heavy = Datamunge::XGBoostClassifier.new(iris, "Species ~ Petal.Length + Petal.Width", 100, 0.3, 6, 50.0)
puts "\nlambda=1 (default):   training accuracy=#{100 * model.training_accuracy}%  deviance=#{model.training_deviance.last}"
puts "lambda=50 (heavy L2): training accuracy=#{100 * heavy.training_accuracy}%  deviance=#{heavy.training_deviance.last}"
heavy.plot_decision_regions("Petal.Length", "Petal.Width").save("xgboost_iris_decision_regions_heavy_lambda.svg")
puts "Saved xgboost_iris_decision_regions_heavy_lambda.svg"
