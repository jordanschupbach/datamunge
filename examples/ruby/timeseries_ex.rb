require "octruby"

def gauss
  Math.sqrt(-2.0 * Math.log(rand)) * Math.cos(2.0 * Math::PI * rand)
end

puts "=================== ARIMA(1,1,1) on a simulated random walk with drift ==================="
srand(42)
y = []
level = 100.0
prev_shock = 0.0
150.times do
  shock = gauss
  level += 0.3 + 0.4 * prev_shock + shock  # integrated ARMA(0,1) increments with drift
  y << level
  prev_shock = shock
end

options = Datamunge::ARIMAOptions.new
options.p = 1
options.d = 1
options.q = 1
options.de_population_size = 80
options.de_max_generations = 400
model = Datamunge::ARIMA.new(y, options)

puts "AR coefficient: #{model.ar_coefficients[0]}"
puts "MA coefficient: #{model.ma_coefficients[0]}"
puts "sigma^2: #{model.sigma2}, AIC: #{model.aic}, BIC: #{model.bic}"

pair = model.forecast_with_intervals(6)
point = pair.first.to_a
se = pair.second.to_a
puts "6-step forecast: #{point}"
puts "forecast std. errors: #{se}"

puts "\n=================== SARIMA(1,0,0)(1,1,0)_12 on a seasonal series ===================\n"
srand(7)
s = []
prev = 0.0
(0...120).each do |i|
  prev = 0.5 * prev + gauss
  s << 20.0 + 0.2 * i + 5.0 * Math.sin(2.0 * Math::PI * i / 12.0) + prev
end

options2 = Datamunge::ARIMAOptions.new
options2.p = 1
options2.seasonal_p = 1
options2.seasonal_d = 1
options2.seasonal_period = 12
options2.de_population_size = 100
options2.de_max_generations = 500
model2 = Datamunge::ARIMA.new(s, options2)
puts "AR coefficient: #{model2.ar_coefficients[0]}, seasonal AR coefficient: #{model2.seasonal_ar_coefficients[0]}"
puts "12-step forecast: #{model2.forecast(12).to_a}"

puts "\n=================== Exponential smoothing: Holt-Winters on retail-style seasonal data ===================\n"
seasonal_shape = [0.8, 0.75, 0.9, 0.95, 1.0, 1.05, 1.1, 1.05, 1.0, 1.1, 1.3, 1.6]
srand(11)
y2 = []
(0...48).each do |i|
  lvl = 100.0 + 2.0 * i
  y2 << lvl * seasonal_shape[i % 12] + gauss * 3.0
end

es_options = Datamunge::ExponentialSmoothingOptions.new
es_options.trend = Datamunge::TrendType_Additive
es_options.seasonal = Datamunge::SeasonalType_Multiplicative
es_options.seasonal_period = 12
es_options.de_population_size = 60
es_options.de_max_generations = 300
es_model = Datamunge::ExponentialSmoothing.new(y2, es_options)
puts "alpha=#{es_model.alpha} beta=#{es_model.beta} gamma=#{es_model.gamma}"
puts "sigma^2: #{es_model.sigma2}, AIC: #{es_model.aic}"
puts "12-month forecast: #{es_model.forecast(12).to_a}"

puts "\n=================== Simple exponential smoothing vs. Holt's linear trend ===================\n"
flat = [50.2, 49.8, 50.5, 49.6, 50.1, 50.3, 49.9, 50.0, 50.4, 49.7]

ses = Datamunge::ExponentialSmoothing.new(flat, Datamunge::ExponentialSmoothingOptions.new)
puts "SES alpha= #{ses.alpha}"
puts "SES 5-step forecast: #{ses.forecast(5).to_a}"

holt_options = Datamunge::ExponentialSmoothingOptions.new
holt_options.trend = Datamunge::TrendType_Additive
holt = Datamunge::ExponentialSmoothing.new(flat, holt_options)
puts "Holt alpha=#{holt.alpha} beta=#{holt.beta}"
puts "Holt 5-step forecast: #{holt.forecast(5).to_a}"
