// CMD: just TARGET=elastic_net_iris_ex run-java
// Ridge, Lasso and Elastic Net on iris, with coefficient-path and CV-curve figures.
import datamunge.ElasticNet;
import datamunge.Lasso;
import datamunge.RPlot;
import datamunge.Ridge;
import datamunge.data.Data;

public class ElasticNetIrisEx {
  static void savePlots(String name, String label, RPlot path, RPlot cv) {
    path.save("elastic_net_" + name + "_path.svg");
    cv.save("elastic_net_" + name + "_cv.svg");
    System.out.println(
        "  " + label + " elastic_net_" + name + "_path.svg, elastic_net_" + name + "_cv.svg");
  }

  public static void main(String[] args) {
    String formula =
        "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width + Sepal.Length:Petal.Width";

    var iris = Data.iris();
    System.out.println("iris: " + iris.nrows() + " rows x " + iris.ncols() + " cols");
    System.out.println("formula: " + formula + "\n");

    // summary() returns the same text printSummary() writes, but goes through System.out,
    // so it stays in order with println when stdout is redirected.
    System.out.println("=================== Ridge ===================");
    var ridge = new Ridge(iris, formula);
    System.out.print(ridge.summary());

    System.out.println("\n=================== Lasso ===================");
    var lasso = new Lasso(iris, formula);
    System.out.print(lasso.summary());

    System.out.println("\n================= Elastic Net =================");
    // new ElasticNet(data, formula, alpha, lambda, nLambda, cvFolds, standardize, seed)
    var elastic = new ElasticNet(iris, formula, 0.5);
    System.out.print(elastic.summary());

    System.out.println(
        "\nSaved figures showing how each model's coefficients respond to the regularization"
            + " strength, and the cross-validation curve used to pick it:");

    savePlots("ridge", "ridge:      ", ridge.plotCoefficientPath(), ridge.plotCvCurve());
    savePlots("lasso", "lasso:      ", lasso.plotCoefficientPath(), lasso.plotCvCurve());
    savePlots(
        "elasticnet", "elastic net:", elastic.plotCoefficientPath(), elastic.plotCvCurve());
  }
}
