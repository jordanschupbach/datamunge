local dm = require("datamunge")

local iris = dm.DataFrame.iris()
print("iris: " .. iris:nrows() .. " rows x " .. iris:ncols() .. " cols\n")

local model = dm.DecisionTreeClassifier(iris, "Species ~ Petal.Length + Petal.Width")
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

model:plot_classification(iris, "Petal.Length", "Petal.Width"):save("tree_iris_classification.svg")
model:plot_decision_regions("Petal.Length", "Petal.Width"):save("tree_iris_decision_regions.svg")
print("\nSaved tree_iris_classification.svg and tree_iris_decision_regions.svg")

-- A shallower tree, for comparison, showing a coarser (but still fairly accurate) decision boundary.
local shallow = dm.DecisionTreeClassifier(iris, "Species ~ Petal.Length + Petal.Width", 2)
print("\nDepth-2 tree training accuracy: " .. (shallow:training_accuracy() * 100.0) .. "% (" .. shallow:leaf_count() .. " leaves)")
shallow:plot_decision_regions("Petal.Length", "Petal.Width"):save("tree_iris_decision_regions_depth2.svg")
print("Saved tree_iris_decision_regions_depth2.svg")
