local dm = require("datamunge")

-- Note: unlike the C++ example, this doesn't call drop_nulls() first (that method isn't
-- exposed on the SWIG-bound DataFrame facade) -- NaiveBayesClassifier drops incomplete rows
-- for its own formula columns internally when fitting, matching every other formula-based
-- model in this library.
local penguins = dm.DataFrame.penguins()
print("penguins: " .. penguins:nrows() .. " rows x " .. penguins:ncols() .. " cols\n")

-- A mix of numeric (Gaussian likelihood) and categorical (frequency-table likelihood)
-- predictors in one formula -- each modeled independently given the class, per the naive
-- Bayes assumption.
local model = dm.NaiveBayesClassifier(penguins, "species ~ bill_length_mm + bill_depth_mm + island + sex")
model:print_summary()

print("\nConfusion matrix (rows = actual, cols = predicted):")
print(model:confusion_matrix():to_string())

print("\nMisclassified rows:")
local predictions = model:predict(penguins)
local misclassified = 0
for i = 0, penguins:nrows() - 1 do
  if not (penguins:is_null("species", i) or penguins:is_null("bill_length_mm", i) or
          penguins:is_null("bill_depth_mm", i) or penguins:is_null("island", i) or penguins:is_null("sex", i)) then
    local actual = penguins:string_at("species", i)
    if predictions[i] ~= actual then
      misclassified = misclassified + 1
      print("  row " .. i .. ": bill_length=" .. penguins:numeric_at("bill_length_mm", i) ..
            " bill_depth=" .. penguins:numeric_at("bill_depth_mm", i) ..
            " island=" .. penguins:string_at("island", i) ..
            " sex=" .. penguins:string_at("sex", i) ..
            "  actual=" .. actual .. "  predicted=" .. predictions[i])
    end
  end
end
print(misclassified .. " misclassified (of " .. penguins:nrows() .. " rows, some incomplete)")

-- plot_decision_regions requires exactly two NUMERIC predictors, so build a separate
-- two-predictor model (bill measurements alone) just for visualization.
local bill_only = dm.NaiveBayesClassifier(penguins, "species ~ bill_length_mm + bill_depth_mm")
print("\nbill-measurements-only model training accuracy: " .. (bill_only:training_accuracy() * 100.0) .. "%")
bill_only:plot_classification(penguins, "bill_length_mm", "bill_depth_mm"):save(
  "naive_bayes_penguins_classification.svg")
bill_only:plot_decision_regions("bill_length_mm", "bill_depth_mm"):save(
  "naive_bayes_penguins_decision_regions.svg")
print("Saved naive_bayes_penguins_classification.svg and naive_bayes_penguins_decision_regions.svg")
