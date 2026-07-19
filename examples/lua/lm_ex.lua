local dm = require("datamunge")

local function dv(t)
  local v = dm.DVector(#t)
  for i, x in ipairs(t) do v[i - 1] = x end
  return v
end

local function sv(t)
  local v = dm.SVector(#t)
  for i, x in ipairs(t) do v[i - 1] = x end
  return v
end

local hp = {110.0, 110.0, 93.0, 110.0, 175.0, 105.0, 245.0, 62.0, 95.0, 123.0}
local wt = {2.62, 2.875, 2.32, 3.215, 3.44, 3.46, 3.57, 3.19, 3.15, 3.44}
local transmission = {"manual", "manual", "manual", "automatic", "automatic",
                       "automatic", "automatic", "automatic", "automatic", "automatic"}
local mpg = {21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2}

local cars = dm.DataFrame()
cars:add_numeric_column("hp", dv(hp))
cars:add_numeric_column("wt", dv(wt))
cars:add_string_column("transmission", sv(transmission))
cars:add_numeric_column("mpg", dv(mpg))

print("Fitting: mpg ~ hp + wt + transmission\n")
local model = dm.LM(cars, "mpg ~ hp + wt + transmission")
model:print_summary()

print("\nSequential ANOVA:")
print(model:anova():to_string())

local newcars = dm.DataFrame()
newcars:add_numeric_column("hp", dv({150.0, 90.0}))
newcars:add_numeric_column("wt", dv({3.0, 2.5}))
newcars:add_string_column("transmission", sv({"manual", "automatic"}))

local frame = model:predict_frame(newcars, "confidence")
print("\nPredictions with 95% confidence intervals:")
print(frame:to_string())

model:save_diagnostic_plots("lm_ex_diagnostics")
print("\nSaved diagnostic plots as lm_ex_diagnostics_*.svg")
