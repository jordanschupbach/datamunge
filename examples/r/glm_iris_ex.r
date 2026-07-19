# See examples/r/lm_ex.r for notes on the flat ClassName_method(obj, ...) call convention.
# Note: simplified relative to the C++ example -- this skips manually building the
# predicted-probability sigmoid-curve ScatterPlot (a long sequence of chained Plot/ScatterPlot
# calls that doesn't translate cleanly to R's non-chaining flat call style) and instead just
# saves the standard GLM diagnostic plots, same as lm_ex.r does for LM.
library(datamunger)

iris <- DataFrame_iris()
n <- DataFrame_nrows(iris)

# Logistic regression: versicolor vs virginica only -- setosa is perfectly separable from the
# other two on these predictors, which sends logistic regression's coefficients toward
# +/-infinity (a genuine degeneracy of the method, not a bug) -- so this is the well-behaved
# binary split for a demo.
is_virginica <- c()
petal_length <- c()
petal_width <- c()
for (i in 0:(n - 1)) {
  species <- DataFrame_string_at(iris, "Species", i)
  if (!(species %in% c("versicolor", "virginica"))) next
  is_virginica <- c(is_virginica, if (species == "virginica") 1.0 else 0.0)
  petal_length <- c(petal_length, DataFrame_numeric_at(iris, "Petal.Length", i))
  petal_width <- c(petal_width, DataFrame_numeric_at(iris, "Petal.Width", i))
}

sub <- DataFrame_empty()
DataFrame_add_numeric_column(sub, "Petal.Length", petal_length)
DataFrame_add_numeric_column(sub, "Petal.Width", petal_width)
DataFrame_add_numeric_column(sub, "is_virginica", is_virginica)

cat("=================== Logistic regression (binomial, logit link) ===================\n")
logit <- GLM(sub, "is_virginica ~ Petal.Length + Petal.Width", "binomial")
GLM_print_summary(logit)

fitted <- GLM_fitted_values(logit)
correct <- sum((fitted >= 0.5) == (is_virginica >= 0.5))
cat("\nResubstitution accuracy at 0.5 threshold:", 100.0 * correct / length(is_virginica), "%\n")

GLM_save_diagnostic_plots(logit, "glm_logistic_iris")
cat("\nSaved glm_logistic_iris_{residuals_vs_fitted,normal_qq,scale_location,residuals_vs_leverage}.svg\n")

# Poisson regression, for contrast: same IRLS engine, different family/link.
cat("\n=================== Poisson regression (log link) ===================\n")
sepal_width <- c()
all_petal_length <- c()
count <- c()
for (i in 0:(n - 1)) {
  sepal_width <- c(sepal_width, DataFrame_numeric_at(iris, "Sepal.Width", i))
  all_petal_length <- c(all_petal_length, DataFrame_numeric_at(iris, "Petal.Length", i))
  count <- c(count, round(DataFrame_numeric_at(iris, "Sepal.Length", i)))
}
count_data <- DataFrame_empty()
DataFrame_add_numeric_column(count_data, "Sepal.Width", sepal_width)
DataFrame_add_numeric_column(count_data, "Petal.Length", all_petal_length)
DataFrame_add_numeric_column(count_data, "count", count)

poisson <- GLM(count_data, "count ~ Sepal.Width + Petal.Length", "poisson")
GLM_print_summary(poisson)
