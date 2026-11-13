# CMD: nix develop .#r --command env R_LIBS_USER=build/r/library Rscript examples/r/timeseries_ex.r
# REPL: nix develop .#r --command env R_LIBS_USER=build/r/library R --no-save --quiet
# Enum-valued options fields (TrendType, SeasonalType) take the enumerator name,
# e.g. options$trend <- "Additive" (or TrendType$Additive).
library(datamunger)

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

options <- ARIMAOptions$new()
options$p <- 1
options$d <- 1
options$q <- 1
options$de_population_size <- 80
options$de_max_generations <- 400
model <- ARIMA$new(y, options)

cat("AR coefficient:", model$ar_coefficients()[1], "\n")
cat("MA coefficient:", model$ma_coefficients()[1], "\n")
cat("sigma^2:", model$sigma2(), ", AIC:", model$aic(), ", BIC:", model$bic(), "\n")

pair <- model$forecast_with_intervals(6)
point <- pair$first
se <- pair$second
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

options2 <- ARIMAOptions$new()
options2$p <- 1
options2$seasonal_p <- 1
options2$seasonal_d <- 1
options2$seasonal_period <- 12
options2$de_population_size <- 100
options2$de_max_generations <- 500
model2 <- ARIMA$new(s, options2)
cat("AR coefficient:", model2$ar_coefficients()[1], ", seasonal AR coefficient:",
    model2$seasonal_ar_coefficients()[1], "\n")
cat("12-step forecast:", model2$forecast(12), "\n")

cat("\n=================== Exponential smoothing: Holt-Winters on retail-style seasonal data ===================\n")
seasonal_shape <- c(0.8, 0.75, 0.9, 0.95, 1.0, 1.05, 1.1, 1.05, 1.0, 1.1, 1.3, 1.6)
set.seed(11)
y2 <- c()
for (i in 0:47) {
  lvl <- 100.0 + 2.0 * i
  y2 <- c(y2, lvl * seasonal_shape[(i %% 12) + 1] + rnorm(1, 0.0, 3.0))
}

es_options <- ExponentialSmoothingOptions$new()
es_options$trend <- "Additive"
es_options$seasonal <- "Multiplicative"
es_options$seasonal_period <- 12
es_options$de_population_size <- 60
es_options$de_max_generations <- 300
es_model <- ExponentialSmoothing$new(y2, es_options)
cat("alpha=", es_model$alpha(), " beta=", es_model$beta(),
    " gamma=", es_model$gamma(), "\n", sep = "")
cat("sigma^2:", es_model$sigma2(), ", AIC:", es_model$aic(), "\n")
cat("12-month forecast:", es_model$forecast(12), "\n")

cat("\n=================== Simple exponential smoothing vs. Holt's linear trend ===================\n")
flat <- c(50.2, 49.8, 50.5, 49.6, 50.1, 50.3, 49.9, 50.0, 50.4, 49.7)

ses <- ExponentialSmoothing$new(flat, ExponentialSmoothingOptions$new())
cat("SES alpha=", ses$alpha(), "\n")
cat("SES 5-step forecast:", ses$forecast(5), "\n")

holt_options <- ExponentialSmoothingOptions$new()
holt_options$trend <- "Additive"
holt <- ExponentialSmoothing$new(flat, holt_options)
cat("Holt alpha=", holt$alpha(), " beta=", holt$beta(), "\n", sep = "")
cat("Holt 5-step forecast:", holt$forecast(5), "\n")
