# CMD: nix develop .#r --command Rscript examples/r/elastic_net_iris_ex.r
library(datamunger)

FORMULA <- "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width + Sepal.Length:Petal.Width"

iris <- DataFrame$iris()
cat(iris$nrows(), "rows x", iris$ncols(), "cols\n")
cat("formula:", FORMULA, "\n\n")

cat("=================== Ridge ===================\n")
ridge <- Ridge$new(iris, FORMULA)
ridge$print_summary()

cat("\n=================== Lasso ===================\n")
lasso <- Lasso$new(iris, FORMULA)
lasso$print_summary()

cat("\n================= Elastic Net =================\n")
# ElasticNet(data, formula, alpha, lambda, n_lambda, cv_folds, standardize, seed)
elastic <- ElasticNet$new(iris, FORMULA, 0.5)
elastic$print_summary()

cat("\nSaved figures showing how each model's coefficients respond to the ",
    "regularization strength, and the cross-validation curve used to pick it:\n")

ridge$plot_coefficient_path()$save("elastic_net_ridge_path.svg")
ridge$plot_cv_curve()$save("elastic_net_ridge_cv.svg")
cat("  ridge:       elastic_net_ridge_path.svg, elastic_net_ridge_cv.svg\n")

lasso$plot_coefficient_path()$save("elastic_net_lasso_path.svg")
lasso$plot_cv_curve()$save("elastic_net_lasso_cv.svg")
cat("  lasso:       elastic_net_lasso_path.svg, elastic_net_lasso_cv.svg\n")

elastic$plot_coefficient_path()$save("elastic_net_elasticnet_path.svg")
elastic$plot_cv_curve()$save("elastic_net_elasticnet_cv.svg")
cat("  elastic net: elastic_net_elasticnet_path.svg, elastic_net_elasticnet_cv.svg\n")
