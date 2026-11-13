# CMD: nix develop .#r --command env R_LIBS_USER=build/r/library Rscript examples/r/gaussian_process_regression_iris_ex.r
# REPL: nix develop .#r --command env R_LIBS_USER=build/r/library R --no-save --quiet
library(datamunger)

FORMULA <- "Petal.Length ~ Petal.Width"

iris <- data$iris()
cat(iris$nrows(), "rows x", iris$ncols(), "cols\n")
cat("formula:", FORMULA, "\n\n")

# Both the length scale and the noise ratio are auto-selected by maximizing the exact log
# marginal likelihood.
model <- GaussianProcessRegression$new(iris, FORMULA)
model$print_summary()

model$plot_fit(iris)$save("gpr_iris_fit.svg")
model$plot_length_scale_profile()$save("gpr_iris_length_scale_profile.svg")
cat("\nSaved gpr_iris_fit.svg and gpr_iris_length_scale_profile.svg\n")

# Unlike every other regressor in this suite, a GP gives a genuine posterior confidence
# interval at every point -- including far outside the training data.
query <- DataFrame$empty()
query$add_numeric_column("Petal.Width", c(0.2, 1.3, 2.5, 10.0))
detail <- model$predict_frame(query, "confidence")
cat("\nPredictions with 95% confidence intervals:\n")
cat(detail$to_string(), "\n")
cat("(Petal.Width=10.0 is far outside the training range [0.1, 2.5] -- note how much wider its\n",
    " interval is than the in-range predictions.)\n")
