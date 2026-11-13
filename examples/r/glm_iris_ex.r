# CMD: nix develop .#r --command env R_LIBS_USER=build/r/library Rscript examples/r/glm_iris_ex.r
# REPL: nix develop .#r --command env R_LIBS_USER=build/r/library R --no-save --quiet
# Note: simplified relative to the C++ example -- this skips manually building the
# predicted-probability sigmoid-curve RPlot and instead just saves the standard GLM
# diagnostic plots, same as lm_ex.r does for LM.
library(datamunger)

iris <- data$iris()
n <- iris$nrows()

# Logistic regression: versicolor vs virginica only -- setosa is perfectly separable from the
# other two on these predictors, which sends logistic regression's coefficients toward
# +/-infinity (a genuine degeneracy of the method, not a bug) -- so this is the well-behaved
# binary split for a demo.
is_virginica <- c()
petal_length <- c()
petal_width <- c()
for (i in 0:(n - 1)) {
  species <- iris$string_at("Species", i)
  if (!(species %in% c("versicolor", "virginica"))) next
  is_virginica <- c(is_virginica, if (species == "virginica") 1.0 else 0.0)
  petal_length <- c(petal_length, iris$numeric_at("Petal.Length", i))
  petal_width <- c(petal_width, iris$numeric_at("Petal.Width", i))
}

sub <- DataFrame$empty()
sub$add_numeric_column("Petal.Length", petal_length)
sub$add_numeric_column("Petal.Width", petal_width)
sub$add_numeric_column("is_virginica", is_virginica)

cat("=================== Logistic regression (binomial, logit link) ===================\n")
logit <- GLM$new(sub, "is_virginica ~ Petal.Length + Petal.Width", "binomial")
logit$print_summary()

fitted <- logit$fitted_values()
correct <- sum((fitted >= 0.5) == (is_virginica >= 0.5))
cat("\nResubstitution accuracy at 0.5 threshold:", 100.0 * correct / length(is_virginica), "%\n")

logit$save_diagnostic_plots("glm_logistic_iris")
cat("\nSaved glm_logistic_iris_{residuals_vs_fitted,normal_qq,scale_location,residuals_vs_leverage}.svg\n")

# Poisson regression, for contrast: same IRLS engine, different family/link.
cat("\n=================== Poisson regression (log link) ===================\n")
sepal_width <- c()
all_petal_length <- c()
count <- c()
for (i in 0:(n - 1)) {
  sepal_width <- c(sepal_width, iris$numeric_at("Sepal.Width", i))
  all_petal_length <- c(all_petal_length, iris$numeric_at("Petal.Length", i))
  count <- c(count, round(iris$numeric_at("Sepal.Length", i)))
}
count_data <- DataFrame$empty()
count_data$add_numeric_column("Sepal.Width", sepal_width)
count_data$add_numeric_column("Petal.Length", all_petal_length)
count_data$add_numeric_column("count", count)

poisson <- GLM$new(count_data, "count ~ Sepal.Width + Petal.Length", "poisson")
poisson$print_summary()
