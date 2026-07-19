local dm = require("datamunge")

local function dv(t)
  local v = dm.DVector(#t)
  for i, x in ipairs(t) do v[i - 1] = x end
  return v
end

local FORMULA = "Petal.Length ~ Petal.Width"

local iris = dm.DataFrame.iris()
print("iris: " .. iris:nrows() .. " rows x " .. iris:ncols() .. " cols")
print("formula: " .. FORMULA .. "\n")

-- Both the length scale and the noise ratio are auto-selected by maximizing the exact log
-- marginal likelihood.
local model = dm.GaussianProcessRegression(iris, FORMULA)
model:print_summary()

model:plot_fit(iris):save("gpr_iris_fit.svg")
model:plot_length_scale_profile():save("gpr_iris_length_scale_profile.svg")
print("\nSaved gpr_iris_fit.svg and gpr_iris_length_scale_profile.svg")

-- Unlike every other regressor in this suite, a GP gives a genuine posterior confidence
-- interval at every point -- including far outside the training data, where it should widen
-- substantially as the model's uncertainty grows.
local query = dm.DataFrame()
query:add_numeric_column("Petal.Width", dv({0.2, 1.3, 2.5, 10.0}))
local detail = model:predict_frame(query, "confidence")
print("\nPredictions with 95% confidence intervals:")
print(detail:to_string())
print("(Petal.Width=10.0 is far outside the training range [0.1, 2.5] -- note how much wider its\n" ..
      " interval is than the in-range predictions.)")
