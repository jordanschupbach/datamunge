// CMD: just TARGET=elastic_net_iris_ex run-csharp
#:project ../../src/datamungecsharp/Datamunge.csproj
// Ridge, Lasso and Elastic Net on iris, with coefficient-path and CV-curve figures.
using Datamunge;

const string formula = "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width + Sepal.Length:Petal.Width";

var iris = Data.Iris()!;
Console.WriteLine($"iris: {iris.Nrows()} rows x {iris.Ncols()} cols");
Console.WriteLine($"formula: {formula}\n");

Console.WriteLine("=================== Ridge ===================");
var ridge = new Ridge(iris, formula);
ridge.PrintSummary();

Console.WriteLine("\n=================== Lasso ===================");
var lasso = new Lasso(iris, formula);
lasso.PrintSummary();

Console.WriteLine("\n================= Elastic Net =================");
// new ElasticNet(data, formula, alpha=0.5, lambda, nLambda, cvFolds, standardize, seed)
var elastic = new ElasticNet(iris, formula, 0.5);
elastic.PrintSummary();

Console.WriteLine("\nSaved figures showing how each model's coefficients respond to the regularization strength, " +
                  "and the cross-validation curve used to pick it:");

(RPlot Path, RPlot Cv, string Name, string Label)[] figures =
[
    (ridge.PlotCoefficientPath(), ridge.PlotCvCurve(), "ridge", "ridge:      "),
    (lasso.PlotCoefficientPath(), lasso.PlotCvCurve(), "lasso", "lasso:      "),
    (elastic.PlotCoefficientPath(), elastic.PlotCvCurve(), "elasticnet", "elastic net:"),
];
foreach (var (path, cv, name, label) in figures)
{
    path.Save($"elastic_net_{name}_path.svg");
    cv.Save($"elastic_net_{name}_cv.svg");
    Console.WriteLine($"  {label} elastic_net_{name}_path.svg, elastic_net_{name}_cv.svg");
}
