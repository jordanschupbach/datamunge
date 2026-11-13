# CMD: nix develop .#r --command env R_LIBS_USER=build/r/library Rscript examples/r/lm_ex.r
# REPL: nix develop .#r --command env R_LIBS_USER=build/r/library R --no-save --quiet
# Classes are R6 generators: construct with Class$new(...), call obj$method(...), read and
# assign fields with obj$field. Static functions live on the generator (data$iris()).
library(datamunger)


hp <- c(110, 110, 93, 110, 175, 105, 245, 62, 95, 123)
wt <- c(2.62, 2.875, 2.32, 3.215, 3.44, 3.46, 3.57, 3.19, 3.15, 3.44)
transmission <- c("manual", "manual", "manual", "automatic", "automatic",
                   "automatic", "automatic", "automatic", "automatic", "automatic")
mpg <- c(21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2)

cars <- DataFrame$empty()
cars$add_numeric_column("hp", hp)
cars$add_numeric_column("wt", wt)
cars$add_string_column("transmission", transmission)
cars$add_numeric_column("mpg", mpg)

cat("Fitting: mpg ~ hp + wt + transmission\n\n")
model <- LM$new(cars, "mpg ~ hp + wt + transmission")
model$print_summary()

cat("\nSequential ANOVA:\n")
cat(model$anova()$to_string(), "\n")

newcars <- DataFrame$empty()
newcars$add_numeric_column("hp", c(150, 90))
newcars$add_numeric_column("wt", c(3.0, 2.5))
newcars$add_string_column("transmission", c("manual", "automatic"))

cat("\nPredictions with 95% confidence intervals:\n")
cat(model$predict_frame(newcars, "confidence")$to_string(), "\n")

model$save_diagnostic_plots("lm_ex_diagnostics")
cat("\nSaved diagnostic plots as lm_ex_diagnostics_*.svg\n")
