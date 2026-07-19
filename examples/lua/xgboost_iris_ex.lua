local dm = require("datamunge")

local iris = dm.DataFrame.iris()
print("iris: " .. iris:nrows() .. " rows x " .. iris:ncols() .. " cols\n")

local model = dm.XGBoostClassifier(iris, "Species ~ Petal.Length + Petal.Width")
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

model:plot_training_deviance():save("xgboost_iris_training_deviance.svg")
model:plot_decision_regions("Petal.Length", "Petal.Width"):save("xgboost_iris_decision_regions.svg")
print("\nSaved xgboost_iris_training_deviance.svg and xgboost_iris_decision_regions.svg")

-- XGBoost's defining lever is regularization: a heavily L2-regularized model (large lambda)
-- grows the same deep trees but keeps every leaf weight small, producing a much smoother
-- decision boundary than the lightly-regularized default.
-- XGBoostClassifier(data, formula, n_trees, learning_rate, max_depth, lambda, alpha, gamma,
--                    min_child_weight, min_samples_leaf, subsample, colsample_bytree, seed)
local heavy = dm.XGBoostClassifier(iris, "Species ~ Petal.Length + Petal.Width", 100, 0.3, 6, 50.0)
local model_dev = model:training_deviance()
local heavy_dev = heavy:training_deviance()
print("\nlambda=1 (default):   training accuracy=" .. (model:training_accuracy() * 100.0) ..
      "%  deviance=" .. model_dev[model_dev:size() - 1])
print("lambda=50 (heavy L2): training accuracy=" .. (heavy:training_accuracy() * 100.0) ..
      "%  deviance=" .. heavy_dev[heavy_dev:size() - 1])
heavy:plot_decision_regions("Petal.Length", "Petal.Width"):save("xgboost_iris_decision_regions_heavy_lambda.svg")
print("Saved xgboost_iris_decision_regions_heavy_lambda.svg")
