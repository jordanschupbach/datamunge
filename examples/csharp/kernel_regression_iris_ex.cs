// CMD: just TARGET=kernel_regression_iris_ex run-csharp
#:project ../../src/datamungecsharp/Datamunge.csproj
// Nadaraya-Watson kernel regression on iris: CV-selected bandwidth vs. too-small and too-large ones.
using Datamunge;

const string formula = "Petal.Length ~ Petal.Width";

var iris = Data.Iris()!;
Console.WriteLine($"iris: {iris.Nrows()} rows x {iris.Ncols()} cols");
Console.WriteLine($"formula: {formula}\n");

var model = new KernelRegression(iris, formula);  // bandwidth auto-selected via leave-one-out CV
model.PrintSummary();

model.PlotFit(iris).Save("kernel_regression_iris_fit.svg");
model.PlotCvCurve().Save("kernel_regression_iris_cv.svg");
Console.WriteLine("\nSaved kernel_regression_iris_fit.svg and kernel_regression_iris_cv.svg");

// A too-small and a too-large bandwidth, for comparison -- the classic kernel regression
// bias-variance story. new KernelRegression(data, formula, kernel, bandwidth, nBandwidth, standardize)
var small = new KernelRegression(iris, formula, "gaussian", 0.05);
Console.WriteLine($"\nbandwidth=0.05 (too small): LOO R-squared={small.RSquared()}  LOO RMSE={small.Rmse()}");
small.PlotFit(iris).Save("kernel_regression_iris_fit_small_bandwidth.svg");

var large = new KernelRegression(iris, formula, "gaussian", 5.0);
Console.WriteLine($"bandwidth=5.0 (too large):  LOO R-squared={large.RSquared()}  LOO RMSE={large.Rmse()}");
large.PlotFit(iris).Save("kernel_regression_iris_fit_large_bandwidth.svg");

Console.WriteLine($"bandwidth={model.Bandwidth()} (CV-selected): LOO R-squared={model.RSquared()}  LOO RMSE={model.Rmse()}");
Console.WriteLine("\nSaved kernel_regression_iris_fit_small_bandwidth.svg and kernel_regression_iris_fit_large_bandwidth.svg");
