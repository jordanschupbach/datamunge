# CMD: just TARGET=naive_bayes_penguins_ex run-ruby
# Naive Bayes on penguins: mixed numeric/categorical predictors, confusion matrix, misclassified rows, plots.
require "datamunge"

# Note: NaiveBayesClassifier drops incomplete rows for its own formula columns internally
# when fitting, matching every other formula-based model in this library.
penguins = Datamunge::Data.penguins
puts "penguins: #{penguins.nrows} rows x #{penguins.ncols} cols\n\n"

# A mix of numeric (Gaussian likelihood) and categorical (frequency-table likelihood)
# predictors in one formula -- each modeled independently given the class, per the naive
# Bayes assumption.
model = Datamunge::NaiveBayesClassifier.new(penguins, "species ~ bill_length_mm + bill_depth_mm + island + sex")
model.print_summary

puts "\nConfusion matrix (rows = actual, cols = predicted):\n#{model.confusion_matrix}"

puts "\nMisclassified rows:"
predictions = model.predict(penguins)
misclassified = 0
columns = %w[species bill_length_mm bill_depth_mm island sex]
penguins.nrows.times do |i|
  next if columns.any? { |c| penguins.is_null(c, i) }
  actual, predicted = penguins.string_at("species", i), predictions[i]
  next if predicted == actual
  misclassified += 1
  puts "  row #{i}: bill_length=#{penguins.numeric_at("bill_length_mm", i)} bill_depth=#{penguins.numeric_at("bill_depth_mm", i)} " \
       "island=#{penguins.string_at("island", i)} sex=#{penguins.string_at("sex", i)}  actual=#{actual}  predicted=#{predicted}"
end
puts "#{misclassified} misclassified (of #{penguins.nrows} rows, some incomplete)"

# plot_decision_regions requires exactly two NUMERIC predictors, so build a separate
# two-predictor model (bill measurements alone) just for visualization.
bill_only = Datamunge::NaiveBayesClassifier.new(penguins, "species ~ bill_length_mm + bill_depth_mm")
puts "\nbill-measurements-only model training accuracy: #{100 * bill_only.training_accuracy}%"
bill_only.plot_classification(penguins, "bill_length_mm", "bill_depth_mm").save("naive_bayes_penguins_classification.svg")
bill_only.plot_decision_regions("bill_length_mm", "bill_depth_mm").save("naive_bayes_penguins_decision_regions.svg")
puts "Saved naive_bayes_penguins_classification.svg and naive_bayes_penguins_decision_regions.svg"
