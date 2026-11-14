# CMD: just TARGET=glm_iris_ex run-julia

# Generalized linear models on iris: logistic regression (binomial) and Poisson regression.
using Datamunge

const iris = Datamunge.Data.iris()
const rows = 0:nrows(iris)-1

# Logistic regression: versicolor vs virginica only -- setosa is perfectly separable from
# the other two on these predictors, which sends logistic regression's coefficients toward
# +/-infinity (a genuine degeneracy of the method, not a bug) -- so this is the well-behaved
# binary split for a demo.
const kept = [i for i in rows if string_at(iris, "Species", i) in ("versicolor", "virginica")]
const is_virginica = [string_at(iris, "Species", i) == "virginica" ? 1.0 : 0.0 for i in kept]
const petal_length = [numeric_at(iris, "Petal.Length", i) for i in kept]
const petal_width = [numeric_at(iris, "Petal.Width", i) for i in kept]

const sub = let frame = DataFrame()
    add_numeric_column(frame, "Petal.Length", petal_length)
    add_numeric_column(frame, "Petal.Width", petal_width)
    add_numeric_column(frame, "is_virginica", is_virginica)
    frame
end

println("=================== Logistic regression (binomial, logit link) ===================")
const logit = GLM(sub, "is_virginica ~ Petal.Length + Petal.Width", "binomial")
print_summary(logit)

let fitted = fitted_values(logit)
    n = length(is_virginica)
    correct = count(i -> (fitted[i] >= 0.5) == (is_virginica[i] >= 0.5), 1:n)
    println("\nResubstitution accuracy at 0.5 threshold: $(100 * correct / n)%")
end

save_diagnostic_plots(logit, "glm_logistic_iris")
println("\nSaved glm_logistic_iris_{residuals_vs_fitted,normal_qq,scale_location,residuals_vs_leverage}.svg")

# Predicted-probability curve across Petal.Length, with Petal.Width held at its mean -- the
# classic sigmoid shape of a fitted logistic regression, with a 95% confidence band.
let width_mean = sum(petal_width) / length(petal_width), grid_n = 100
    pl_min, pl_max = minimum(petal_length) - 0.3, maximum(petal_length) + 0.3
    grid = DataFrame()
    add_numeric_column(grid, "Petal.Length", [pl_min + (pl_max - pl_min) * i / (grid_n - 1) for i in 0:grid_n-1])
    add_numeric_column(grid, "Petal.Width", fill(width_mean, grid_n))
    curve_frame = predict_frame(logit, grid, "confidence")
    println("\nPredicted-probability curve (first 5 rows):")
    println(to_string(curve_frame, 5))
end

# Poisson regression, for contrast: same IRLS engine, different family/link.
println("\n=================== Poisson regression (log link) ===================")
let count_data = DataFrame()
    add_numeric_column(count_data, "Sepal.Width", [numeric_at(iris, "Sepal.Width", i) for i in rows])
    add_numeric_column(count_data, "Petal.Length", [numeric_at(iris, "Petal.Length", i) for i in rows])
    add_numeric_column(count_data, "count", [floor(numeric_at(iris, "Sepal.Length", i) + 0.5) for i in rows])
    print_summary(GLM(count_data, "count ~ Sepal.Width + Petal.Length", "poisson"))
end
