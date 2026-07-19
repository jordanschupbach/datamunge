# See examples/r/lm_ex.r for notes on the flat ClassName_method(obj, ...) call convention.
library(datamunger)

FORMULA <- "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width + Sepal.Length:Petal.Width"

iris <- DataFrame_iris()
cat(DataFrame_nrows(iris), "rows x", DataFrame_ncols(iris), "cols\n")
cat("formula:", FORMULA, "\n\n")

cat("=================== Ridge ===================\n")
ridge <- Ridge(iris, FORMULA)
Ridge_print_summary(ridge)

cat("\n=================== Lasso ===================\n")
lasso <- Lasso(iris, FORMULA)
Lasso_print_summary(lasso)

cat("\n================= Elastic Net =================\n")
# ElasticNet(data, formula, alpha, lambda, n_lambda, cv_folds, standardize, seed)
elastic <- ElasticNet(iris, FORMULA, 0.5)
ElasticNet_print_summary(elastic)

cat("\nSaved figures showing how each model's coefficients respond to the ",
    "regularization strength, and the cross-validation curve used to pick it:\n")

Plot_save(Ridge_plot_coefficient_path(ridge), "elastic_net_ridge_path.svg")
Plot_save(Ridge_plot_cv_curve(ridge), "elastic_net_ridge_cv.svg")
cat("  ridge:       elastic_net_ridge_path.svg, elastic_net_ridge_cv.svg\n")

Plot_save(Lasso_plot_coefficient_path(lasso), "elastic_net_lasso_path.svg")
Plot_save(Lasso_plot_cv_curve(lasso), "elastic_net_lasso_cv.svg")
cat("  lasso:       elastic_net_lasso_path.svg, elastic_net_lasso_cv.svg\n")

Plot_save(ElasticNet_plot_coefficient_path(elastic), "elastic_net_elasticnet_path.svg")
Plot_save(ElasticNet_plot_cv_curve(elastic), "elastic_net_elasticnet_cv.svg")
cat("  elastic net: elastic_net_elasticnet_path.svg, elastic_net_elasticnet_cv.svg\n")
