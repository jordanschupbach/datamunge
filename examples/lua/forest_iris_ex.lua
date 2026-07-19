local dm = require("datamunge")

local iris = dm.DataFrame.iris()
print("iris: " .. iris:nrows() .. " rows x " .. iris:ncols() .. " cols\n")

local model = dm.RandomForestClassifier(iris, "Species ~ Petal.Length + Petal.Width")
model:print_summary()

print("\nConfusion matrix (rows = actual, cols = predicted):")
print(model:confusion_matrix():to_string())

print("\nMisclassified rows:")
local predictions = model:predict(iris)
local misclassified = 0
for i = 0, iris:nrows() - 1 do
  local actual = iris:string_at("Species", i)
  if predictions[i] ~= actual then
    misclassified = misclassified + 1
    print("  row " .. i .. ": Petal.Length=" .. iris:numeric_at("Petal.Length", i) ..
          " Petal.Width=" .. iris:numeric_at("Petal.Width", i) ..
          "  actual=" .. actual .. "  predicted=" .. predictions[i])
  end
end
print(misclassified .. " of " .. iris:nrows() .. " misclassified (" .. (100.0 * misclassified / iris:nrows()) .. "%)")

model:plot_classification(iris, "Petal.Length", "Petal.Width"):save("forest_iris_classification.svg")
model:plot_decision_regions("Petal.Length", "Petal.Width"):save("forest_iris_decision_regions.svg")
print("\nSaved forest_iris_classification.svg and forest_iris_decision_regions.svg")

-- A small forest, for comparison, showing how out-of-bag accuracy stabilizes as more trees
-- are added -- the defining random forest effect a single decision tree cannot demonstrate.
-- RandomForestClassifier(data, formula, n_trees, max_depth, min_samples_split, min_samples_leaf,
--                         max_features, criterion, bootstrap, sample_fraction, seed)
local small_forest = dm.RandomForestClassifier(iris, "Species ~ Petal.Length + Petal.Width", 5)
print("\n5-tree forest:   training accuracy=" .. (small_forest:training_accuracy() * 100.0) ..
      "%  OOB accuracy=" .. (small_forest:oob_accuracy() * 100.0) .. "%")
print("100-tree forest: training accuracy=" .. (model:training_accuracy() * 100.0) ..
      "%  OOB accuracy=" .. (model:oob_accuracy() * 100.0) .. "%")
small_forest:plot_decision_regions("Petal.Length", "Petal.Width"):save("forest_iris_decision_regions_5trees.svg")
print("Saved forest_iris_decision_regions_5trees.svg")
