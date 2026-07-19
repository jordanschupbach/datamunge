local dm = require("datamunge")

local function dv(t)
  local v = dm.DVector(#t)
  for i, x in ipairs(t) do v[i - 1] = x end
  return v
end

local function to_table(vec)
  local out = {}
  for i = 0, vec:size() - 1 do out[i + 1] = vec[i] end
  return out
end

local function gauss()
  local u1, u2 = math.random(), math.random()
  return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
end

print("=================== ARIMA(1,1,1) on a simulated random walk with drift ===================")
math.randomseed(42)
local y = {}
local level = 100.0
local prev_shock = 0.0
for _ = 1, 150 do
  local shock = gauss()
  level = level + 0.3 + 0.4 * prev_shock + shock  -- integrated ARMA(0,1) increments with drift
  table.insert(y, level)
  prev_shock = shock
end

local options = dm.ARIMAOptions()
options.p = 1
options.d = 1
options.q = 1
options.de_population_size = 80
options.de_max_generations = 400
local model = dm.ARIMA(dv(y), options)

print("AR coefficient: " .. model:ar_coefficients()[0])
print("MA coefficient: " .. model:ma_coefficients()[0])
print("sigma^2: " .. model:sigma2() .. ", AIC: " .. model:aic() .. ", BIC: " .. model:bic())

local pair = model:forecast_with_intervals(6)
print("6-step forecast: " .. table.concat(to_table(pair.first), ", "))
print("forecast std. errors: " .. table.concat(to_table(pair.second), ", "))

print("\n=================== SARIMA(1,0,0)(1,1,0)_12 on a seasonal series ===================")
math.randomseed(7)
local s = {}
local prev = 0.0
for i = 0, 119 do
  prev = 0.5 * prev + gauss()
  table.insert(s, 20.0 + 0.2 * i + 5.0 * math.sin(2.0 * math.pi * i / 12.0) + prev)
end

local options2 = dm.ARIMAOptions()
options2.p = 1
options2.seasonal_p = 1
options2.seasonal_d = 1
options2.seasonal_period = 12
options2.de_population_size = 100
options2.de_max_generations = 500
local model2 = dm.ARIMA(dv(s), options2)
print("AR coefficient: " .. model2:ar_coefficients()[0] .. ", seasonal AR coefficient: " .. model2:seasonal_ar_coefficients()[0])
print("12-step forecast: " .. table.concat(to_table(model2:forecast(12)), ", "))

print("\n=================== Exponential smoothing: Holt-Winters on retail-style seasonal data ===================")
local seasonal_shape = {0.8, 0.75, 0.9, 0.95, 1.0, 1.05, 1.1, 1.05, 1.0, 1.1, 1.3, 1.6}
math.randomseed(11)
local y2 = {}
for i = 0, 47 do
  local lvl = 100.0 + 2.0 * i
  table.insert(y2, lvl * seasonal_shape[(i % 12) + 1] + gauss() * 3.0)
end

local es_options = dm.ExponentialSmoothingOptions()
es_options.trend = dm.TrendType_Additive
es_options.seasonal = dm.SeasonalType_Multiplicative
es_options.seasonal_period = 12
es_options.de_population_size = 60
es_options.de_max_generations = 300
local es_model = dm.ExponentialSmoothing(dv(y2), es_options)
print("alpha=" .. es_model:alpha() .. " beta=" .. es_model:beta() .. " gamma=" .. es_model:gamma())
print("sigma^2: " .. es_model:sigma2() .. ", AIC: " .. es_model:aic())
print("12-month forecast: " .. table.concat(to_table(es_model:forecast(12)), ", "))

print("\n=================== Simple exponential smoothing vs. Holt's linear trend ===================")
local flat = {50.2, 49.8, 50.5, 49.6, 50.1, 50.3, 49.9, 50.0, 50.4, 49.7}

local ses = dm.ExponentialSmoothing(dv(flat), dm.ExponentialSmoothingOptions())
print("SES alpha= " .. ses:alpha())
print("SES 5-step forecast: " .. table.concat(to_table(ses:forecast(5)), ", "))

local holt_options = dm.ExponentialSmoothingOptions()
holt_options.trend = dm.TrendType_Additive
local holt = dm.ExponentialSmoothing(dv(flat), holt_options)
print("Holt alpha=" .. holt:alpha() .. " beta=" .. holt:beta())
print("Holt 5-step forecast: " .. table.concat(to_table(holt:forecast(5)), ", "))
