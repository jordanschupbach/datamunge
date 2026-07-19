local dm = require("datamunge")

local function dv(t)
  local v = dm.DVector(#t)
  for i, x in ipairs(t) do v[i - 1] = x end
  return v
end

local iris = dm.DataFrame.iris()

-- Logistic regression: versicolor vs virginica only -- setosa is perfectly separable from
-- the other two on these predictors, which sends logistic regression's coefficients toward
-- +/-infinity (a genuine degeneracy of the method, not a bug) -- so this is the well-behaved
-- binary split for a demo.
local is_virginica, petal_length, petal_width = {}, {}, {}
for i = 0, iris:nrows() - 1 do
  local species = iris:string_at("Species", i)
  if species == "versicolor" or species == "virginica" then
    table.insert(is_virginica, species == "virginica" and 1.0 or 0.0)
    table.insert(petal_length, iris:numeric_at("Petal.Length", i))
    table.insert(petal_width, iris:numeric_at("Petal.Width", i))
  end
end

local sub = dm.DataFrame()
sub:add_numeric_column("Petal.Length", dv(petal_length))
sub:add_numeric_column("Petal.Width", dv(petal_width))
sub:add_numeric_column("is_virginica", dv(is_virginica))

print("=================== Logistic regression (binomial, logit link) ===================")
local logit = dm.GLM(sub, "is_virginica ~ Petal.Length + Petal.Width", "binomial")
logit:print_summary()

local fitted = logit:fitted_values()
local correct = 0
for i = 0, #is_virginica - 1 do
  if (fitted[i] >= 0.5) == (is_virginica[i + 1] >= 0.5) then correct = correct + 1 end
end
print("\nResubstitution accuracy at 0.5 threshold: " .. (100.0 * correct / #is_virginica) .. "%")

logit:save_diagnostic_plots("glm_logistic_iris")
print("\nSaved glm_logistic_iris_{residuals_vs_fitted,normal_qq,scale_location,residuals_vs_leverage}.svg")

-- Predicted-probability curve across Petal.Length, with Petal.Width held at its mean -- the
-- classic sigmoid shape of a fitted logistic regression, with a 95% confidence band.
local width_sum = 0
for _, w in ipairs(petal_width) do width_sum = width_sum + w end
local width_mean = width_sum / #petal_width
local grid_n = 100
local pl_min, pl_max = petal_length[1], petal_length[1]
for _, x in ipairs(petal_length) do
  if x < pl_min then pl_min = x end
  if x > pl_max then pl_max = x end
end
pl_min = pl_min - 0.3
pl_max = pl_max + 0.3
local grid_x = {}
local width_col = {}
for i = 0, grid_n - 1 do
  grid_x[i + 1] = pl_min + (pl_max - pl_min) * i / (grid_n - 1)
  width_col[i + 1] = width_mean
end
local grid = dm.DataFrame()
grid:add_numeric_column("Petal.Length", dv(grid_x))
grid:add_numeric_column("Petal.Width", dv(width_col))
local curve_frame = logit:predict_frame(grid, "confidence")
print("\nPredicted-probability curve (first 5 rows):")
print(curve_frame:to_string(5))

-- Poisson regression, for contrast: same IRLS engine, different family/link.
print("\n=================== Poisson regression (log link) ===================")
local count = {}
local sepal_width = {}
local all_petal_length = {}
for i = 0, iris:nrows() - 1 do
  count[i + 1] = math.floor(iris:numeric_at("Sepal.Length", i) + 0.5) * 1.0
  sepal_width[i + 1] = iris:numeric_at("Sepal.Width", i)
  all_petal_length[i + 1] = iris:numeric_at("Petal.Length", i)
end
local count_data = dm.DataFrame()
count_data:add_numeric_column("Sepal.Width", dv(sepal_width))
count_data:add_numeric_column("Petal.Length", dv(all_petal_length))
count_data:add_numeric_column("count", dv(count))

local poisson = dm.GLM(count_data, "count ~ Sepal.Width + Petal.Length", "poisson")
poisson:print_summary()
