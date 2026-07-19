using System;

class Program {
  static void Main() {
    const string FORMULA = "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width + Sepal.Length:Petal.Width";

    var iris = DataFrame.iris();
    Console.WriteLine($"iris: {iris.nrows()} rows x {iris.ncols()} cols");
    Console.WriteLine($"formula: {FORMULA}\n");

    Console.WriteLine("=================== Ridge ===================");
    var ridge = new Ridge(iris, FORMULA);
    ridge.print_summary();

    Console.WriteLine("\n=================== Lasso ===================");
    var lasso = new Lasso(iris, FORMULA);
    lasso.print_summary();

    Console.WriteLine("\n================= Elastic Net =================");
    var elastic = new ElasticNet(iris, FORMULA, 0.5);
    elastic.print_summary();

    Console.WriteLine("\nSaved figures showing how each model's coefficients respond to the regularization strength, and the cross-validation curve used to pick it:");

    ridge.plot_coefficient_path().save("elastic_net_ridge_path.svg");
    ridge.plot_cv_curve().save("elastic_net_ridge_cv.svg");
    Console.WriteLine("  ridge:       elastic_net_ridge_path.svg, elastic_net_ridge_cv.svg");

    lasso.plot_coefficient_path().save("elastic_net_lasso_path.svg");
    lasso.plot_cv_curve().save("elastic_net_lasso_cv.svg");
    Console.WriteLine("  lasso:       elastic_net_lasso_path.svg, elastic_net_lasso_cv.svg");

    elastic.plot_coefficient_path().save("elastic_net_elasticnet_path.svg");
    elastic.plot_cv_curve().save("elastic_net_elasticnet_cv.svg");
    Console.WriteLine("  elastic net: elastic_net_elasticnet_path.svg, elastic_net_elasticnet_cv.svg");
  }
}
