module app;

import std.stdio : writeln;
import datamunge;

void main() {
  string FORMULA = "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width + Sepal.Length:Petal.Width";

  auto iris = DataFrame.iris();
  writeln("iris: ", iris.nrows(), " rows x ", iris.ncols(), " cols");
  writeln("formula: ", FORMULA, "\n");

  writeln("=================== Ridge ===================");
  auto ridge = new Ridge(iris, FORMULA);
  ridge.print_summary();

  writeln("\n=================== Lasso ===================");
  auto lasso = new Lasso(iris, FORMULA);
  lasso.print_summary();

  writeln("\n================= Elastic Net =================");
  // ElasticNet(data, formula, alpha, lambda, n_lambda, cv_folds, standardize, seed)
  auto elastic = new ElasticNet(iris, FORMULA, 0.5);
  elastic.print_summary();

  writeln("\nSaved figures showing how each model's coefficients respond to the " ~
          "regularization strength, and the cross-validation curve used to pick it:");

  ridge.plot_coefficient_path().save("elastic_net_ridge_path.svg");
  ridge.plot_cv_curve().save("elastic_net_ridge_cv.svg");
  writeln("  ridge:       elastic_net_ridge_path.svg, elastic_net_ridge_cv.svg");

  lasso.plot_coefficient_path().save("elastic_net_lasso_path.svg");
  lasso.plot_cv_curve().save("elastic_net_lasso_cv.svg");
  writeln("  lasso:       elastic_net_lasso_path.svg, elastic_net_lasso_cv.svg");

  elastic.plot_coefficient_path().save("elastic_net_elasticnet_path.svg");
  elastic.plot_cv_curve().save("elastic_net_elasticnet_cv.svg");
  writeln("  elastic net: elastic_net_elasticnet_path.svg, elastic_net_elasticnet_cv.svg");
}
