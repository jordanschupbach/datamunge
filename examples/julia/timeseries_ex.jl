# CMD: just TARGET=timeseries_ex run-julia

# Time series: ARIMA, seasonal ARIMA and exponential smoothing (SES, Holt, Holt-Winters).
using Datamunge
using Random

rng = Xoshiro(42)
seed!(n) = (global rng = Xoshiro(n))
uniform() = 1 - rand(rng)  # (0, 1]: avoids log(0) in the Box-Muller transform.
gauss() = sqrt(-2 * log(uniform())) * cos(2π * uniform())
joined(v) = join(v, ", ")

println("=================== ARIMA(1,1,1) on a simulated random walk with drift ===================")
seed!(42)
const y = let level = 100.0, prev_shock = 0.0, out = Float64[]
    for _ in 1:150
        shock = gauss()
        level += 0.3 + 0.4 * prev_shock + shock  # integrated ARMA(0,1) increments with drift
        push!(out, level)
        prev_shock = shock
    end
    out
end

const model = let options = ARIMAOptions()
    options.p, options.d, options.q = 1, 1, 1
    options.de_population_size = 80
    options.de_max_generations = 400
    ARIMA(y, options)
end

println("AR coefficient: $(ar_coefficients(model)[1])")
println("MA coefficient: $(ma_coefficients(model)[1])")
println("sigma^2: $(sigma2(model)), AIC: $(aic(model)), BIC: $(bic(model))")

let (forecast_values, std_errors) = forecast_with_intervals(model, 6)
    println("6-step forecast: $(joined(forecast_values))")
    println("forecast std. errors: $(joined(std_errors))")
end

println("\n=================== SARIMA(1,0,0)(1,1,0)_12 on a seasonal series ===================")
seed!(7)
const s = let prev = 0.0, out = Float64[]
    for i in 0:119
        prev = 0.5 * prev + gauss()
        push!(out, 20 + 0.2 * i + 5 * sin(2π * i / 12) + prev)
    end
    out
end

const model2 = let options = ARIMAOptions()
    options.p = 1
    options.seasonal_p = 1
    options.seasonal_d = 1
    options.seasonal_period = 12
    options.de_population_size = 100
    options.de_max_generations = 500
    ARIMA(s, options)
end
println("AR coefficient: $(ar_coefficients(model2)[1]), seasonal AR coefficient: $(seasonal_ar_coefficients(model2)[1])")
println("12-step forecast: $(joined(forecast(model2, 12)))")

println("\n=================== Exponential smoothing: Holt-Winters on retail-style seasonal data ===================")
const seasonal_shape = [0.8, 0.75, 0.9, 0.95, 1.0, 1.05, 1.1, 1.05, 1.0, 1.1, 1.3, 1.6]
seed!(11)
const y2 = [(100 + 2i) * seasonal_shape[i % 12 + 1] + gauss() * 3 for i in 0:47]

const es_model = let options = ExponentialSmoothingOptions()
    options.trend = :Additive
    options.seasonal = :Multiplicative
    options.seasonal_period = 12
    options.de_population_size = 60
    options.de_max_generations = 300
    ExponentialSmoothing(y2, options)
end
println("alpha=$(alpha(es_model)) beta=$(beta(es_model)) gamma=$(gamma(es_model))")
println("sigma^2: $(sigma2(es_model)), AIC: $(aic(es_model))")
println("12-month forecast: $(joined(forecast(es_model, 12)))")

println("\n=================== Simple exponential smoothing vs. Holt's linear trend ===================")
const flat = [50.2, 49.8, 50.5, 49.6, 50.1, 50.3, 49.9, 50.0, 50.4, 49.7]

let ses = ExponentialSmoothing(flat, ExponentialSmoothingOptions())
    println("SES alpha= $(alpha(ses))")
    println("SES 5-step forecast: $(joined(forecast(ses, 5)))")
end

let holt_options = ExponentialSmoothingOptions()
    holt_options.trend = :Additive
    holt = ExponentialSmoothing(flat, holt_options)
    println("Holt alpha=$(alpha(holt)) beta=$(beta(holt))")
    println("Holt 5-step forecast: $(joined(forecast(holt, 5)))")
end
