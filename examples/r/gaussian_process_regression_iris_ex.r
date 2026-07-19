# See examples/r/lm_ex.r for notes on the flat ClassName_method(obj, ...) call convention.
library(datamunger)

FORMULA <- "Petal.Length ~ Petal.Width"

iris <- DataFrame_iris()
cat(DataFrame_nrows(iris), "rows x", DataFrame_ncols(iris), "cols\n")
cat("formula:", FORMULA, "\n\n")

# Both the length scale and the noise ratio are auto-selected by maximizing the exact log
# marginal likelihood.
model <- GaussianProcessRegression(iris, FORMULA)
GaussianProcessRegression_print_summary(model)

Plot_save(GaussianProcessRegression_plot_fit(model, iris), "gpr_iris_fit.svg")
Plot_save(GaussianProcessRegression_plot_length_scale_profile(model), "gpr_iris_length_scale_profile.svg")
cat("\nSaved gpr_iris_fit.svg and gpr_iris_length_scale_profile.svg\n")

# Unlike every other regressor in this suite, a GP gives a genuine posterior confidence
# interval at every point -- including far outside the training data.
query <- DataFrame_empty()
DataFrame_add_numeric_column(query, "Petal.Width", c(0.2, 1.3, 2.5, 10.0))
detail <- GaussianProcessRegression_predict_frame(model, query, "confidence")
cat("\nPredictions with 95% confidence intervals:\n")
cat(DataFrame_to_string(detail), "\n")
cat("(Petal.Width=10.0 is far outside the training range [0.1, 2.5] -- note how much wider its\n",
    " interval is than the in-range predictions.)\n")
