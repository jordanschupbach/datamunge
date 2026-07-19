# Note: this uses the flat ClassName_method(obj, ...) call form throughout, not R's usual
# obj$method(...) syntax -- the latter unreliably fails for overloaded methods (anything with
# a default argument) in this package's generated R bindings. See memory/
# datamunge_r_dollar_dispatch_bug.md for the diagnosis; DataFrame_empty() is used instead of
# the bare DataFrame() constructor for the same reason (a separate, related R-backend bug).
# String *vector* columns also can't be passed as a real SVector object (the generated
# wrapper tries to as.character() it and fails) -- add_string_column_encoded() with a simple
# length-prefixed, unit-separator-delimited encoding sidesteps the whole marshaling problem.
library(datamunger)

encode_strings <- function(values) {
  paste0(length(values), "\x1e", paste(values, collapse = "\x1f"))
}

hp <- c(110, 110, 93, 110, 175, 105, 245, 62, 95, 123)
wt <- c(2.62, 2.875, 2.32, 3.215, 3.44, 3.46, 3.57, 3.19, 3.15, 3.44)
transmission <- c("manual", "manual", "manual", "automatic", "automatic",
                   "automatic", "automatic", "automatic", "automatic", "automatic")
mpg <- c(21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2)

cars <- DataFrame_empty()
DataFrame_add_numeric_column(cars, "hp", hp)
DataFrame_add_numeric_column(cars, "wt", wt)
DataFrame_add_string_column_encoded(cars, "transmission", encode_strings(transmission))
DataFrame_add_numeric_column(cars, "mpg", mpg)

cat("Fitting: mpg ~ hp + wt + transmission\n\n")
model <- LM(cars, "mpg ~ hp + wt + transmission")
LM_print_summary(model)

cat("\nSequential ANOVA:\n")
cat(DataFrame_to_string(LM_anova(model)), "\n")

newcars <- DataFrame_empty()
DataFrame_add_numeric_column(newcars, "hp", c(150, 90))
DataFrame_add_numeric_column(newcars, "wt", c(3.0, 2.5))
DataFrame_add_string_column_encoded(newcars, "transmission", encode_strings(c("manual", "automatic")))

cat("\nPredictions with 95% confidence intervals:\n")
cat(DataFrame_to_string(LM_predict_frame(model, newcars, "confidence")), "\n")

LM_save_diagnostic_plots(model, "lm_ex_diagnostics")
cat("\nSaved diagnostic plots as lm_ex_diagnostics_*.svg\n")
