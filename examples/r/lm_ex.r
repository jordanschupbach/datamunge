library(datamunger)

hp <- c(110, 110, 93, 110, 175, 105, 245, 62, 95, 123)
wt <- c(2.62, 2.875, 2.32, 3.215, 3.44, 3.46, 3.57, 3.19, 3.15, 3.44)
transmission <- c("manual", "manual", "manual", "automatic", "automatic",
                   "automatic", "automatic", "automatic", "automatic", "automatic")
mpg <- c(21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2)

cars <- DataFrame()
cars$add_column("hp", hp)
cars$add_column("wt", wt)
cars$add_column("transmission", transmission)
cars$add_column("mpg", mpg)

cat("Fitting: mpg ~ hp + wt + transmission\n\n")
model <- LM(cars, "mpg ~ hp + wt + transmission")
model$print_summary()

cat("\nSequential ANOVA:\n")
cat(model$anova()$to_string(), "\n")

newcars <- DataFrame()
newcars$add_column("hp", c(150, 90))
newcars$add_column("wt", c(3.0, 2.5))
newcars$add_column("transmission", c("manual", "automatic"))

cat("\nPredictions with 95% confidence intervals:\n")
cat(model$predict_frame(newcars, "confidence")$to_string(), "\n")

model$save_diagnostic_plots("lm_ex_diagnostics")
cat("\nSaved diagnostic plots as lm_ex_diagnostics_*.svg\n")
