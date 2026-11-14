# CMD: just TARGET=timeseries_ex run-ruby
# Time series: ARIMA, seasonal ARIMA and exponential smoothing (SES, Holt, Holt-Winters).
require "datamunge"

rng = Random.new(42)
# Box-Muller on (0, 1] uniforms (avoids log(0)); reassigning `rng` reseeds it.
gauss = -> { Math.sqrt(-2 * Math.log(1 - rng.rand)) * Math.cos(2 * Math::PI * (1 - rng.rand)) }

puts "=================== ARIMA(1,1,1) on a simulated random walk with drift ==================="
rng = Random.new(42)
level = 100.0
prev_shock = 0.0
y = Array.new(150) do
  shock = gauss.()
  level += 0.3 + 0.4 * prev_shock + shock  # integrated ARMA(0,1) increments with drift
  prev_shock = shock
  level
end

options = Datamunge::ARIMAOptions.new.tap do |o|
  o.p_ = 1  # `p` clashes with Kernel#p
  o.d = 1
  o.q = 1
  o.de_population_size = 80
  o.de_max_generations = 400
end
model = Datamunge::ARIMA.new(y, options)

puts "AR coefficient: #{model.ar_coefficients[0]}"
puts "MA coefficient: #{model.ma_coefficients[0]}"
puts "sigma^2: #{model.sigma2}, AIC: #{model.aic}, BIC: #{model.bic}"

forecast_values, std_errors = model.forecast_with_intervals(6)
puts "6-step forecast: #{forecast_values.join(", ")}"
puts "forecast std. errors: #{std_errors.join(", ")}"

puts "\n=================== SARIMA(1,0,0)(1,1,0)_12 on a seasonal series ==================="
rng = Random.new(7)
prev = 0.0
s = Array.new(120) do |i|
  prev = 0.5 * prev + gauss.()
  20 + 0.2 * i + 5 * Math.sin(2 * Math::PI * i / 12) + prev
end

model2 = Datamunge::ARIMA.new(s, Datamunge::ARIMAOptions.new.tap do |o|
  o.p_ = 1
  o.seasonal_p = 1
  o.seasonal_d = 1
  o.seasonal_period = 12
  o.de_population_size = 100
  o.de_max_generations = 500
end)
puts "AR coefficient: #{model2.ar_coefficients[0]}, seasonal AR coefficient: #{model2.seasonal_ar_coefficients[0]}"
puts "12-step forecast: #{model2.forecast(12).join(", ")}"

puts "\n=================== Exponential smoothing: Holt-Winters on retail-style seasonal data ==================="
seasonal_shape = [0.8, 0.75, 0.9, 0.95, 1.0, 1.05, 1.1, 1.05, 1.0, 1.1, 1.3, 1.6]
rng = Random.new(11)
y2 = Array.new(48) { |i| (100 + 2 * i) * seasonal_shape[i % 12] + gauss.() * 3 }

es_model = Datamunge::ExponentialSmoothing.new(y2, Datamunge::ExponentialSmoothingOptions.new.tap do |o|
  o.trend = Datamunge::TrendType::ADDITIVE
  o.seasonal = Datamunge::SeasonalType::MULTIPLICATIVE
  o.seasonal_period = 12
  o.de_population_size = 60
  o.de_max_generations = 300
end)
puts "alpha=#{es_model.alpha} beta=#{es_model.beta} gamma=#{es_model.gamma}"
puts "sigma^2: #{es_model.sigma2}, AIC: #{es_model.aic}"
puts "12-month forecast: #{es_model.forecast(12).join(", ")}"

puts "\n=================== Simple exponential smoothing vs. Holt's linear trend ==================="
flat = [50.2, 49.8, 50.5, 49.6, 50.1, 50.3, 49.9, 50.0, 50.4, 49.7]

ses = Datamunge::ExponentialSmoothing.new(flat, Datamunge::ExponentialSmoothingOptions.new)
puts "SES alpha= #{ses.alpha}"
puts "SES 5-step forecast: #{ses.forecast(5).join(", ")}"

holt = Datamunge::ExponentialSmoothing.new(flat, Datamunge::ExponentialSmoothingOptions.new.tap { |o| o.trend = :Additive })
puts "Holt alpha=#{holt.alpha} beta=#{holt.beta}"
puts "Holt 5-step forecast: #{holt.forecast(5).join(", ")}"
