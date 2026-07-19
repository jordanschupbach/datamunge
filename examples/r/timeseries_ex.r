# See examples/r/lm_ex.r for notes on the flat ClassName_method(obj, ...) call convention.
# Enum-valued options fields (TrendType, SeasonalType) are set by passing the plain string
# name, e.g. ExponentialSmoothingOptions_trend_set(options, "Additive") -- see
# datamunge_r_dollar_dispatch_bug.md.
# DVector instance methods (DVector_size/__getitem__/etc.) are broken through their normal
# wrapper: the generated code does `self = as.numeric(self)` before the .Call, because self's
# type (std::vector<double>) collides with the global vector<double>-parameter %apply typemap
# and the codegen can't tell "self" apart from a regular numeric-vector parameter. Bypass the
# broken wrapper with direct .Call()s instead -- needed only for ARIMA_forecast_with_intervals's
# DVectorPair result below (a genuine vector<double>* result, not a by-value return).
library(datamunger)

dvector_to_r <- function(v) {
  n <- .Call("R_swig_DVector_size", v, FALSE, PACKAGE = "datamunger")
  sapply(0:(n - 1), function(i) .Call("R_swig_DVector___getitem__", v, as.integer(i), FALSE, PACKAGE = "datamunger"))
}

cat("=================== ARIMA(1,1,1) on a simulated random walk with drift ===================\n")
set.seed(42)
y <- c()
level <- 100.0
prev_shock <- 0.0
for (i in 1:150) {
  shock <- rnorm(1, 0.0, 1.0)
  level <- level + 0.3 + 0.4 * prev_shock + shock  # integrated ARMA(0,1) increments with drift
  y <- c(y, level)
  prev_shock <- shock
}

options <- ARIMAOptions()
ARIMAOptions_p_set(options, 1)
ARIMAOptions_d_set(options, 1)
ARIMAOptions_q_set(options, 1)
ARIMAOptions_de_population_size_set(options, 80)
ARIMAOptions_de_max_generations_set(options, 400)
model <- ARIMA(y, options)

cat("AR coefficient:", ARIMA_ar_coefficients(model)[1], "\n")
cat("MA coefficient:", ARIMA_ma_coefficients(model)[1], "\n")
cat("sigma^2:", ARIMA_sigma2(model), ", AIC:", ARIMA_aic(model), ", BIC:", ARIMA_bic(model), "\n")

pair <- ARIMA_forecast_with_intervals(model, 6)
point <- dvector_to_r(DVectorPair_first_get(pair))
se <- dvector_to_r(DVectorPair_second_get(pair))
cat("6-step forecast:", point, "\n")
cat("forecast std. errors:", se, "\n")

cat("\n=================== SARIMA(1,0,0)(1,1,0)_12 on a seasonal series ===================\n")
set.seed(7)
s <- c()
prev <- 0.0
for (i in 0:119) {
  prev <- 0.5 * prev + rnorm(1, 0.0, 1.0)
  s <- c(s, 20.0 + 0.2 * i + 5.0 * sin(2.0 * pi * i / 12.0) + prev)
}

options2 <- ARIMAOptions()
ARIMAOptions_p_set(options2, 1)
ARIMAOptions_seasonal_p_set(options2, 1)
ARIMAOptions_seasonal_d_set(options2, 1)
ARIMAOptions_seasonal_period_set(options2, 12)
ARIMAOptions_de_population_size_set(options2, 100)
ARIMAOptions_de_max_generations_set(options2, 500)
model2 <- ARIMA(s, options2)
cat("AR coefficient:", ARIMA_ar_coefficients(model2)[1], ", seasonal AR coefficient:",
    ARIMA_seasonal_ar_coefficients(model2)[1], "\n")
cat("12-step forecast:", ARIMA_forecast(model2, 12), "\n")

cat("\n=================== Exponential smoothing: Holt-Winters on retail-style seasonal data ===================\n")
seasonal_shape <- c(0.8, 0.75, 0.9, 0.95, 1.0, 1.05, 1.1, 1.05, 1.0, 1.1, 1.3, 1.6)
set.seed(11)
y2 <- c()
for (i in 0:47) {
  lvl <- 100.0 + 2.0 * i
  y2 <- c(y2, lvl * seasonal_shape[(i %% 12) + 1] + rnorm(1, 0.0, 3.0))
}

es_options <- ExponentialSmoothingOptions()
ExponentialSmoothingOptions_trend_set(es_options, "Additive")
ExponentialSmoothingOptions_seasonal_set(es_options, "Multiplicative")
ExponentialSmoothingOptions_seasonal_period_set(es_options, 12)
ExponentialSmoothingOptions_de_population_size_set(es_options, 60)
ExponentialSmoothingOptions_de_max_generations_set(es_options, 300)
es_model <- ExponentialSmoothing(y2, es_options)
cat("alpha=", ExponentialSmoothing_alpha(es_model), " beta=", ExponentialSmoothing_beta(es_model),
    " gamma=", ExponentialSmoothing_gamma(es_model), "\n", sep = "")
cat("sigma^2:", ExponentialSmoothing_sigma2(es_model), ", AIC:", ExponentialSmoothing_aic(es_model), "\n")
cat("12-month forecast:", ExponentialSmoothing_forecast(es_model, 12), "\n")

cat("\n=================== Simple exponential smoothing vs. Holt's linear trend ===================\n")
flat <- c(50.2, 49.8, 50.5, 49.6, 50.1, 50.3, 49.9, 50.0, 50.4, 49.7)

ses <- ExponentialSmoothing(flat, ExponentialSmoothingOptions())
cat("SES alpha=", ExponentialSmoothing_alpha(ses), "\n")
cat("SES 5-step forecast:", ExponentialSmoothing_forecast(ses, 5), "\n")

holt_options <- ExponentialSmoothingOptions()
ExponentialSmoothingOptions_trend_set(holt_options, "Additive")
holt <- ExponentialSmoothing(flat, holt_options)
cat("Holt alpha=", ExponentialSmoothing_alpha(holt), " beta=", ExponentialSmoothing_beta(holt), "\n", sep = "")
cat("Holt 5-step forecast:", ExponentialSmoothing_forecast(holt, 5), "\n")
