# CMD: just TARGET=elastic_net_iris_ex run-julia

# Ridge, Lasso and Elastic Net on iris, with coefficient-path and CV-curve figures.
using Datamunge

const formula = "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width + Sepal.Length:Petal.Width"

const iris = Datamunge.Data.iris()
println("iris: $(nrows(iris)) rows x $(ncols(iris)) cols")
println("formula: $formula\n")

println("=================== Ridge ===================")
const ridge = Ridge(iris, formula)
print_summary(ridge)

println("\n=================== Lasso ===================")
const lasso = Lasso(iris, formula)
print_summary(lasso)

println("\n================= Elastic Net =================")
# ElasticNet(data, formula, alpha=0.5, lambda, n_lambda, cv_folds, standardize, seed)
const elastic = ElasticNet(iris, formula, 0.5)
print_summary(elastic)

println("\nSaved figures showing how each model's coefficients respond to the regularization strength, " *
        "and the cross-validation curve used to pick it:")

for (model, name, label) in ((ridge, "ridge", "ridge:      "), (lasso, "lasso", "lasso:      "),
                             (elastic, "elasticnet", "elastic net:"))
    save(plot_coefficient_path(model), "elastic_net_$(name)_path.svg")
    save(plot_cv_curve(model), "elastic_net_$(name)_cv.svg")
    println("  $label elastic_net_$(name)_path.svg, elastic_net_$(name)_cv.svg")
end
