' CMD: just TARGET=gaussian_process_regression_iris_ex run-vb
' Gaussian process regression of Petal.Length on Petal.Width (iris), with confidence intervals.
Option Strict On
Imports System
Imports Datamunge

Module Example
    Const formula As String = "Petal.Length ~ Petal.Width"

    Sub Main()
        Dim iris = Data.Iris()
        Console.WriteLine($"iris: {iris.Nrows()} rows x {iris.Ncols()} cols")
        Console.WriteLine($"formula: {formula}{vbLf}")

        ' Both the length scale and the noise ratio are auto-selected by maximizing the exact log
        ' marginal likelihood.
        Dim model = New GaussianProcessRegression(iris, formula)
        model.PrintSummary()

        model.PlotFit(iris).Save("gpr_iris_fit.svg")
        model.PlotLengthScaleProfile().Save("gpr_iris_length_scale_profile.svg")
        Console.WriteLine($"{vbLf}Saved gpr_iris_fit.svg and gpr_iris_length_scale_profile.svg")

        ' Unlike every other regressor in this suite, a GP gives a genuine posterior confidence
        ' interval at every point -- including far outside the training data, where it should widen
        ' substantially as the model's uncertainty grows.
        Dim query = New DataFrame()
        query.AddNumericColumn("Petal.Width", {0.2, 1.3, 2.5, 10.0})
        Dim detail = model.PredictFrame(query, "confidence")
        Console.WriteLine($"{vbLf}Predictions with 95% confidence intervals:")
        Console.WriteLine(detail)
        Console.WriteLine($"(Petal.Width=10.0 is far outside the training range [0.1, 2.5] -- note how much wider its{vbLf} interval is than the in-range predictions.)")
    End Sub
End Module
