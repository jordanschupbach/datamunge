from pydatamunge import datamunge as dm

FORMULA = "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width + Sepal.Length:Petal.Width"

iris = dm.DataFrame.iris()
print(f"iris: {iris.nrows()} rows x {iris.ncols()} cols")
print(f"formula: {FORMULA}\n")

print("=================== Ridge ===================")
ridge = dm.Ridge(iris, FORMULA)
ridge.print_summary()

print("\n=================== Lasso ===================")
lasso = dm.Lasso(iris, FORMULA)
lasso.print_summary()

print("\n================= Elastic Net =================")
# ElasticNet(data, formula, alpha, lambda, n_lambda, cv_folds, standardize, seed)
elastic = dm.ElasticNet(iris, FORMULA, 0.5)
elastic.print_summary()

print("\nSaved figures showing how each model's coefficients respond to the "
      "regularization strength, and the cross-validation curve used to pick it:")

ridge.plot_coefficient_path().save("elastic_net_ridge_path.svg")
ridge.plot_cv_curve().save("elastic_net_ridge_cv.svg")
print("  ridge:       elastic_net_ridge_path.svg, elastic_net_ridge_cv.svg")

lasso.plot_coefficient_path().save("elastic_net_lasso_path.svg")
lasso.plot_cv_curve().save("elastic_net_lasso_cv.svg")
print("  lasso:       elastic_net_lasso_path.svg, elastic_net_lasso_cv.svg")

elastic.plot_coefficient_path().save("elastic_net_elasticnet_path.svg")
elastic.plot_cv_curve().save("elastic_net_elasticnet_cv.svg")
print("  elastic net: elastic_net_elasticnet_path.svg, elastic_net_elasticnet_cv.svg")
