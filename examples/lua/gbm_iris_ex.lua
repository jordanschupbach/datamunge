local dm = require("datamunge")

local iris = dm.DataFrame.iris()
print("iris: " .. iris:nrows() .. " rows x " .. iris:ncols() .. " cols\n")

local model = dm.GBMClassifier(iris, "Species ~ Petal.Length + Petal.Width")
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

model:plot_training_deviance():save("gbm_iris_training_deviance.svg")
model:plot_decision_regions("Petal.Length", "Petal.Width"):save("gbm_iris_decision_regions.svg")
print("\nSaved gbm_iris_training_deviance.svg and gbm_iris_decision_regions.svg")

-- A handful of boosting rounds vs a well-boosted ensemble: each additional round chips away
-- at the training loss, gradually sharpening the decision boundary.
-- GBMClassifier(data, formula, n_trees, learning_rate, max_depth, min_samples_split, min_samples_leaf, subsample, seed)
local few = dm.GBMClassifier(iris, "Species ~ Petal.Length + Petal.Width", 5)
local few_dev = few:training_deviance()
local model_dev = model:training_deviance()
print("\n5-round ensemble:   training accuracy=" .. (few:training_accuracy() * 100.0) ..
      "%  deviance=" .. few_dev[few_dev:size() - 1])
print("100-round ensemble: training accuracy=" .. (model:training_accuracy() * 100.0) ..
      "%  deviance=" .. model_dev[model_dev:size() - 1])
few:plot_decision_regions("Petal.Length", "Petal.Width"):save("gbm_iris_decision_regions_5rounds.svg")
print("Saved gbm_iris_decision_regions_5rounds.svg")
