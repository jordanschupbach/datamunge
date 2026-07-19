local dm = require("datamunge")

local function dv(t)
  local v = dm.DVector(#t)
  for i, x in ipairs(t) do v[i - 1] = x end
  return v
end

local iris = dm.DataFrame.iris()
print("iris: " .. iris:nrows() .. " rows x " .. iris:ncols() .. " cols\n")

print("=== RBF kernel (default) ===")
local rbf_model = dm.SVM(iris, "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width")
rbf_model:print_summary()
print("\nConfusion matrix (rows = actual, cols = predicted):")
print(rbf_model:confusion_matrix():to_string())

print("\n=== Linear kernel, for comparison ===")
local linear_model = dm.SVM(iris, "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width", "linear")
print("Training accuracy: " .. (linear_model:training_accuracy() * 100.0) .. "%")
print("Support vectors: " .. linear_model:num_support_vectors())

local newdata = dm.DataFrame()
newdata:add_numeric_column("Sepal.Length", dv({5.1, 6.0, 6.5, 6.2}))
newdata:add_numeric_column("Sepal.Width", dv({3.5, 2.7, 3.0, 2.8}))
newdata:add_numeric_column("Petal.Length", dv({1.4, 4.5, 5.5, 4.8}))
newdata:add_numeric_column("Petal.Width", dv({0.2, 1.5, 2.0, 1.8}))

print("\nRBF predictions for new flowers (votes out of 3 one-vs-one pairs):")
print(rbf_model:predict_frame(newdata):to_string())
