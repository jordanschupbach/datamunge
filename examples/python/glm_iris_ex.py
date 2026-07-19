from pydatamunge import datamunge as dm

iris = dm.DataFrame.iris()

# Logistic regression: versicolor vs virginica only -- setosa is perfectly separable from
# the other two on these predictors, which sends logistic regression's coefficients toward
# +/-infinity (a genuine degeneracy of the method, not a bug) -- so this is the well-behaved
# binary split for a demo.
is_virginica, petal_length, petal_width = [], [], []
for i in range(iris.nrows()):
    species = iris.string_at("Species", i)
    if species not in ("versicolor", "virginica"):
        continue
    is_virginica.append(1.0 if species == "virginica" else 0.0)
    petal_length.append(iris.numeric_at("Petal.Length", i))
    petal_width.append(iris.numeric_at("Petal.Width", i))

sub = dm.DataFrame()
sub.add_numeric_column("Petal.Length", petal_length)
sub.add_numeric_column("Petal.Width", petal_width)
sub.add_numeric_column("is_virginica", is_virginica)

print("=================== Logistic regression (binomial, logit link) ===================")
logit = dm.GLM(sub, "is_virginica ~ Petal.Length + Petal.Width", "binomial")
logit.print_summary()

fitted = logit.fitted_values()
correct = sum(1 for i in range(len(is_virginica)) if (fitted[i] >= 0.5) == (is_virginica[i] >= 0.5))
print(f"\nResubstitution accuracy at 0.5 threshold: {100.0 * correct / len(is_virginica)}%")

logit.save_diagnostic_plots("glm_logistic_iris")
print("\nSaved glm_logistic_iris_{residuals_vs_fitted,normal_qq,scale_location,residuals_vs_leverage}.svg")

# Predicted-probability curve across Petal.Length, with Petal.Width held at its mean -- the
# classic sigmoid shape of a fitted logistic regression, with a 95% confidence band.
width_mean = sum(petal_width) / len(petal_width)
grid_n = 100
pl_min, pl_max = min(petal_length) - 0.3, max(petal_length) + 0.3
grid_x = [pl_min + (pl_max - pl_min) * i / (grid_n - 1) for i in range(grid_n)]
grid = dm.DataFrame()
grid.add_numeric_column("Petal.Length", grid_x)
grid.add_numeric_column("Petal.Width", [width_mean] * grid_n)
curve_frame = logit.predict_frame(grid, "confidence")
print("\nPredicted-probability curve (first 5 rows):")
print(curve_frame.to_string(5))

# Poisson regression, for contrast: same IRLS engine, different family/link.
print("\n=================== Poisson regression (log link) ===================")
count = [round(iris.numeric_at("Sepal.Length", i)) for i in range(iris.nrows())]
count_data = dm.DataFrame()
sepal_width = [iris.numeric_at("Sepal.Width", i) for i in range(iris.nrows())]
all_petal_length = [iris.numeric_at("Petal.Length", i) for i in range(iris.nrows())]
count_data.add_numeric_column("Sepal.Width", sepal_width)
count_data.add_numeric_column("Petal.Length", all_petal_length)
count_data.add_numeric_column("count", [float(c) for c in count])

poisson = dm.GLM(count_data, "count ~ Sepal.Width + Petal.Length", "poisson")
poisson.print_summary()
