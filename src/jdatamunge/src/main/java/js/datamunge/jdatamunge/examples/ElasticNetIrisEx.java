package js.datamunge.jdatamunge.examples;

import js.datamunge.jdatamunge.DataFrame;
import js.datamunge.jdatamunge.Ridge;
import js.datamunge.jdatamunge.Lasso;
import js.datamunge.jdatamunge.ElasticNet;

public class ElasticNetIrisEx {
  static {
    System.loadLibrary("datamunge_jni");
  }

  public static void run() {
    String formula = "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width + Sepal.Length:Petal.Width";

    var iris = DataFrame.iris();
    System.out.println("iris: " + iris.nrows() + " rows x " + iris.ncols() + " cols");
    System.out.println("formula: " + formula + "\n");

    System.out.println("=================== Ridge ===================");
    var ridge = new Ridge(iris, formula);
    ridge.print_summary();

    System.out.println("\n=================== Lasso ===================");
    var lasso = new Lasso(iris, formula);
    lasso.print_summary();

    System.out.println("\n================= Elastic Net =================");
    var elastic = new ElasticNet(iris, formula, 0.5);
    elastic.print_summary();

    System.out.println("\nSaved figures showing how each model's coefficients respond to the regularization strength, and the cross-validation curve used to pick it:");

    ridge.plot_coefficient_path().save("elastic_net_ridge_path.svg");
    ridge.plot_cv_curve().save("elastic_net_ridge_cv.svg");
    System.out.println("  ridge:       elastic_net_ridge_path.svg, elastic_net_ridge_cv.svg");

    lasso.plot_coefficient_path().save("elastic_net_lasso_path.svg");
    lasso.plot_cv_curve().save("elastic_net_lasso_cv.svg");
    System.out.println("  lasso:       elastic_net_lasso_path.svg, elastic_net_lasso_cv.svg");

    elastic.plot_coefficient_path().save("elastic_net_elasticnet_path.svg");
    elastic.plot_cv_curve().save("elastic_net_elasticnet_cv.svg");
    System.out.println("  elastic net: elastic_net_elasticnet_path.svg, elastic_net_elasticnet_cv.svg");
  }
}
