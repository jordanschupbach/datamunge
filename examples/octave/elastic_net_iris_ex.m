1;

datamunge;

FORMULA = "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width + Sepal.Length:Petal.Width";

iris = DataFrame_iris();
printf("iris: %d rows x %d cols\n", DataFrame_nrows(iris), DataFrame_ncols(iris));
printf("formula: %s\n\n", FORMULA);

printf("=================== Ridge ===================\n");
ridge = Ridge(iris, FORMULA);
Ridge_print_summary(ridge);

printf("\n=================== Lasso ===================\n");
lasso = Lasso(iris, FORMULA);
Lasso_print_summary(lasso);

printf("\n================= Elastic Net =================\n");
% ElasticNet(data, formula, alpha, lambda, n_lambda, cv_folds, standardize, seed)
elastic = ElasticNet(iris, FORMULA, 0.5);
ElasticNet_print_summary(elastic);

printf("\nSaved figures showing how each model's coefficients respond to the regularization strength, and the cross-validation curve used to pick it:\n");

Plot_save(Ridge_plot_coefficient_path(ridge), "elastic_net_ridge_path.svg");
Plot_save(Ridge_plot_cv_curve(ridge), "elastic_net_ridge_cv.svg");
printf("  ridge:       elastic_net_ridge_path.svg, elastic_net_ridge_cv.svg\n");

Plot_save(Lasso_plot_coefficient_path(lasso), "elastic_net_lasso_path.svg");
Plot_save(Lasso_plot_cv_curve(lasso), "elastic_net_lasso_cv.svg");
printf("  lasso:       elastic_net_lasso_path.svg, elastic_net_lasso_cv.svg\n");

Plot_save(ElasticNet_plot_coefficient_path(elastic), "elastic_net_elasticnet_path.svg");
Plot_save(ElasticNet_plot_cv_curve(elastic), "elastic_net_elasticnet_cv.svg");
printf("  elastic net: elastic_net_elasticnet_path.svg, elastic_net_elasticnet_cv.svg\n");
