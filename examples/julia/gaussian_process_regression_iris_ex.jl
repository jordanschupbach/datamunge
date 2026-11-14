# CMD: just TARGET=gaussian_process_regression_iris_ex run-julia

# Gaussian process regression of Petal.Length on Petal.Width (iris), with confidence intervals.
using Datamunge

const formula = "Petal.Length ~ Petal.Width"

const iris = Datamunge.Data.iris()
println("iris: $(nrows(iris)) rows x $(ncols(iris)) cols")
println("formula: $formula\n")

# Both the length scale and the noise ratio are auto-selected by maximizing the exact log
# marginal likelihood.
const model = GaussianProcessRegression(iris, formula)
print_summary(model)

save(plot_fit(model, iris), "gpr_iris_fit.svg")
save(plot_length_scale_profile(model), "gpr_iris_length_scale_profile.svg")
println("\nSaved gpr_iris_fit.svg and gpr_iris_length_scale_profile.svg")

# Unlike every other regressor in this suite, a GP gives a genuine posterior confidence
# interval at every point -- including far outside the training data, where it should widen
# substantially as the model's uncertainty grows.
let query = DataFrame()
    add_numeric_column(query, "Petal.Width", [0.2, 1.3, 2.5, 10.0])
    detail = predict_frame(model, query, "confidence")
    println("\nPredictions with 95% confidence intervals:")
    println(to_string(detail))
    println("(Petal.Width=10.0 is far outside the training range [0.1, 2.5] -- note how much wider its\n interval is than the in-range predictions.)")
end
