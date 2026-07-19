local dm = require("datamunge")

local iris = dm.DataFrame.iris()
print("iris: " .. iris:nrows() .. " rows x " .. iris:ncols() .. " cols\n")

local model = dm.KNNClassifier(iris, "Species ~ Petal.Length + Petal.Width")
model:print_summary()

print("\nConfusion matrix (leave-one-out; rows = actual, cols = predicted):")
print(model:confusion_matrix():to_string())

print("\nLeave-one-out misclassified rows:")
local fitted = model:predict(iris)
local misclassified = 0
for i = 0, iris:nrows() - 1 do
  local actual = iris:string_at("Species", i)
  if fitted[i] ~= actual then
    misclassified = misclassified + 1
    print("  row " .. i .. ": Petal.Length=" .. iris:numeric_at("Petal.Length", i) ..
          " Petal.Width=" .. iris:numeric_at("Petal.Width", i) ..
          "  actual=" .. actual .. "  predicted=" .. fitted[i])
  end
end
print(misclassified .. " of " .. iris:nrows() .. " misclassified (" .. (100.0 * misclassified / iris:nrows()) .. "%)")

model:plot_decision_regions("Petal.Length", "Petal.Width"):save("knn_iris_decision_regions_k5.svg")
print("\nSaved knn_iris_decision_regions_k5.svg")

-- k=1 memorizes every training point exactly (jagged, overfit boundary with an island around
-- every point, including noise); k=25 averages over a much larger neighborhood (very smooth,
-- underfit boundary). KNNClassifier(data, formula, k, metric, weighted, standardize)
local k1 = dm.KNNClassifier(iris, "Species ~ Petal.Length + Petal.Width", 1)
print("\nk=1  leave-one-out accuracy: " .. (k1:training_accuracy() * 100.0) .. "%")
k1:plot_decision_regions("Petal.Length", "Petal.Width"):save("knn_iris_decision_regions_k1.svg")
print("Saved knn_iris_decision_regions_k1.svg")

local k25 = dm.KNNClassifier(iris, "Species ~ Petal.Length + Petal.Width", 25)
print("\nk=25 leave-one-out accuracy: " .. (k25:training_accuracy() * 100.0) .. "%")
k25:plot_decision_regions("Petal.Length", "Petal.Width"):save("knn_iris_decision_regions_k25.svg")
print("Saved knn_iris_decision_regions_k25.svg")
