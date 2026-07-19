require "octruby"

# Note: unlike the C++ example, this doesn't call drop_nulls() first (that method isn't
# exposed on the SWIG-bound DataFrame facade) -- NaiveBayesClassifier drops incomplete rows
# for its own formula columns internally when fitting, matching every other formula-based
# model in this library.
penguins = Datamunge::DataFrame.penguins
puts "penguins: #{penguins.nrows} rows x #{penguins.ncols} cols\n\n"

# A mix of numeric (Gaussian likelihood) and categorical (frequency-table likelihood)
# predictors in one formula -- each modeled independently given the class, per the naive
# Bayes assumption.
model = Datamunge::NaiveBayesClassifier.new(penguins, "species ~ bill_length_mm + bill_depth_mm + island + sex")
model.print_summary

puts "\nConfusion matrix (rows = actual, cols = predicted):"
puts model.confusion_matrix.to_string

puts "\nMisclassified rows:"
predictions = model.predict(penguins)
misclassified = 0
penguins.nrows.times do |i|
  next if penguins.is_null("species", i) || penguins.is_null("bill_length_mm", i) ||
          penguins.is_null("bill_depth_mm", i) || penguins.is_null("island", i) || penguins.is_null("sex", i)
  actual = penguins.string_at("species", i)
  next if predictions[i] == actual
  misclassified += 1
  puts "  row #{i}: bill_length=#{penguins.numeric_at('bill_length_mm', i)} " \
       "bill_depth=#{penguins.numeric_at('bill_depth_mm', i)} island=#{penguins.string_at('island', i)} " \
       "sex=#{penguins.string_at('sex', i)}  actual=#{actual}  predicted=#{predictions[i]}"
end
puts "#{misclassified} misclassified (of #{penguins.nrows} rows, some incomplete)"

# plot_decision_regions requires exactly two NUMERIC predictors, so build a separate
# two-predictor model (bill measurements alone) just for visualization.
bill_only = Datamunge::NaiveBayesClassifier.new(penguins, "species ~ bill_length_mm + bill_depth_mm")
puts "\nbill-measurements-only model training accuracy: #{bill_only.training_accuracy * 100.0}%"
bill_only.plot_classification(penguins, "bill_length_mm", "bill_depth_mm").save(
  "naive_bayes_penguins_classification.svg")
bill_only.plot_decision_regions("bill_length_mm", "bill_depth_mm").save(
  "naive_bayes_penguins_decision_regions.svg")
puts "Saved naive_bayes_penguins_classification.svg and naive_bayes_penguins_decision_regions.svg"
