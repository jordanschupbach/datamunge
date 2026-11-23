' CMD: just TARGET=kernel_regression_iris_ex run-vb
' Nadaraya-Watson kernel regression on iris: CV-selected bandwidth vs. too-small and too-large ones.
Option Strict On
Imports System
Imports Datamunge

Module Example
    Const formula As String = "Petal.Length ~ Petal.Width"

    Sub Main()
        Dim iris = Data.Iris()
        Console.WriteLine($"iris: {iris.Nrows()} rows x {iris.Ncols()} cols")
        Console.WriteLine($"formula: {formula}{vbLf}")

        Dim model = New KernelRegression(iris, formula)  ' bandwidth auto-selected via leave-one-out CV
        model.PrintSummary()

        model.PlotFit(iris).Save("kernel_regression_iris_fit.svg")
        model.PlotCvCurve().Save("kernel_regression_iris_cv.svg")
        Console.WriteLine($"{vbLf}Saved kernel_regression_iris_fit.svg and kernel_regression_iris_cv.svg")

        ' A too-small and a too-large bandwidth, for comparison -- the classic kernel regression
        ' bias-variance story. New KernelRegression(data, formula, kernel, bandwidth, nBandwidth, standardize)
        Dim small = New KernelRegression(iris, formula, "gaussian", 0.05)
        Console.WriteLine($"{vbLf}bandwidth=0.05 (too small): LOO R-squared={small.RSquared()}  LOO RMSE={small.Rmse()}")
        small.PlotFit(iris).Save("kernel_regression_iris_fit_small_bandwidth.svg")

        Dim large = New KernelRegression(iris, formula, "gaussian", 5.0)
        Console.WriteLine($"bandwidth=5.0 (too large):  LOO R-squared={large.RSquared()}  LOO RMSE={large.Rmse()}")
        large.PlotFit(iris).Save("kernel_regression_iris_fit_large_bandwidth.svg")

        Console.WriteLine($"bandwidth={model.Bandwidth()} (CV-selected): LOO R-squared={model.RSquared()}  LOO RMSE={model.Rmse()}")
        Console.WriteLine($"{vbLf}Saved kernel_regression_iris_fit_small_bandwidth.svg and kernel_regression_iris_fit_large_bandwidth.svg")
    End Sub
End Module
