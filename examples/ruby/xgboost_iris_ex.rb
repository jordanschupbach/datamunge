require "octruby"

iris = Datamunge::DataFrame.iris
puts "iris: #{iris.nrows} rows x #{iris.ncols} cols\n\n"

model = Datamunge::XGBoostClassifier.new(iris, "Species ~ Petal.Length + Petal.Width")
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

model.plot_training_deviance.save("xgboost_iris_training_deviance.svg")
model.plot_decision_regions("Petal.Length", "Petal.Width").save("xgboost_iris_decision_regions.svg")
puts "\nSaved xgboost_iris_training_deviance.svg and xgboost_iris_decision_regions.svg"

# XGBoost's defining lever is regularization: a heavily L2-regularized model (large lambda)
# grows the same deep trees but keeps every leaf weight small, producing a much smoother
# decision boundary than the lightly-regularized default.
# XGBoostClassifier(data, formula, n_trees, learning_rate, max_depth, lambda, alpha, gamma,
#                    min_child_weight, min_samples_leaf, subsample, colsample_bytree, seed)
heavy = Datamunge::XGBoostClassifier.new(iris, "Species ~ Petal.Length + Petal.Width", 100, 0.3, 6, 50.0)
puts "\nlambda=1 (default):   training accuracy=#{model.training_accuracy * 100.0}%" \
     "  deviance=#{model.training_deviance[-1]}"
puts "lambda=50 (heavy L2): training accuracy=#{heavy.training_accuracy * 100.0}%" \
     "  deviance=#{heavy.training_deviance[-1]}"
heavy.plot_decision_regions("Petal.Length", "Petal.Width").save("xgboost_iris_decision_regions_heavy_lambda.svg")
puts "Saved xgboost_iris_decision_regions_heavy_lambda.svg"
