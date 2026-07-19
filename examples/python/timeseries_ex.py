import math
import random

from pydatamunge import datamunge as dm

print("=================== ARIMA(1,1,1) on a simulated random walk with drift ===================")
rng = random.Random(42)
y = []
level, prev_shock = 100.0, 0.0
for _ in range(150):
    shock = rng.normalvariate(0.0, 1.0)
    level += 0.3 + 0.4 * prev_shock + shock  # integrated ARMA(0,1) increments with drift
    y.append(level)
    prev_shock = shock

options = dm.ARIMAOptions()
options.p, options.d, options.q = 1, 1, 1
options.de_population_size, options.de_max_generations = 80, 400
model = dm.ARIMA(y, options)

print("AR coefficient:", model.ar_coefficients()[0])
print("MA coefficient:", model.ma_coefficients()[0])
print(f"sigma^2: {model.sigma2()}, AIC: {model.aic()}, BIC: {model.bic()}")

point, se = model.forecast_with_intervals(6)
print("6-step forecast:", list(point))
print("forecast std. errors:", list(se))

print("\n=================== SARIMA(1,0,0)(1,1,0)_12 on a seasonal series ===================")
rng = random.Random(7)
s = []
prev = 0.0
for i in range(120):
    prev = 0.5 * prev + rng.normalvariate(0.0, 1.0)
    s.append(20.0 + 0.2 * i + 5.0 * math.sin(2.0 * math.pi * i / 12.0) + prev)

options = dm.ARIMAOptions()
options.p = 1
options.seasonal_p, options.seasonal_d, options.seasonal_period = 1, 1, 12
options.de_population_size, options.de_max_generations = 100, 500
model = dm.ARIMA(s, options)
print("AR coefficient:", model.ar_coefficients()[0], ", seasonal AR coefficient:", model.seasonal_ar_coefficients()[0])
print("12-step forecast:", list(model.forecast(12)))

print("\n=================== Exponential smoothing: Holt-Winters on retail-style seasonal data "
      "===================")
seasonal_shape = [0.8, 0.75, 0.9, 0.95, 1.0, 1.05, 1.1, 1.05, 1.0, 1.1, 1.3, 1.6]
rng = random.Random(11)
y2 = []
for i in range(48):
    lvl = 100.0 + 2.0 * i
    y2.append(lvl * seasonal_shape[i % 12] + rng.normalvariate(0.0, 3.0))

options = dm.ExponentialSmoothingOptions()
options.trend = dm.TrendType_Additive
options.seasonal = dm.SeasonalType_Multiplicative
options.seasonal_period = 12
options.de_population_size, options.de_max_generations = 60, 300
model = dm.ExponentialSmoothing(y2, options)
print(f"alpha={model.alpha()} beta={model.beta()} gamma={model.gamma()}")
print(f"sigma^2: {model.sigma2()}, AIC: {model.aic()}")
print("12-month forecast:", list(model.forecast(12)))

print("\n=================== Simple exponential smoothing vs. Holt's linear trend "
      "===================")
flat = [50.2, 49.8, 50.5, 49.6, 50.1, 50.3, 49.9, 50.0, 50.4, 49.7]

ses = dm.ExponentialSmoothing(flat, dm.ExponentialSmoothingOptions())
print("SES alpha=", ses.alpha())
print("SES 5-step forecast:", list(ses.forecast(5)))

holt_options = dm.ExponentialSmoothingOptions()
holt_options.trend = dm.TrendType_Additive
holt = dm.ExponentialSmoothing(flat, holt_options)
print(f"Holt alpha={holt.alpha()} beta={holt.beta()}")
print("Holt 5-step forecast:", list(holt.forecast(5)))
