' CMD: just TARGET=bspline_ex run-vb
' Tensor-product B-spline regression: fit z ~ bs(x, y) to a smooth surface and predict.
Option Strict On
Imports System
Imports System.Collections.Generic
Imports Datamunge

Module Example
    Sub Main()
        ' Sample a smooth surface z = sin(x) + y^2 on an 8x8 grid over [0, 1]^2.
        Dim x As New List(Of Double), y As New List(Of Double), z As New List(Of Double)
        For iy = 0 To 7
            For ix = 0 To 7
                Dim xv = ix / 7.0, yv = iy / 7.0
                x.Add(xv)
                y.Add(yv)
                z.Add(Math.Sin(xv) + yv * yv)
            Next
        Next
        Dim data = New DataFrame()
        data.AddNumericColumn("x", x)
        data.AddNumericColumn("y", y)
        data.AddNumericColumn("z", z)

        Dim surface = New LM(data, "z ~ bs(x, y)")
        Console.WriteLine("Fitted z ~ bs(x, y)")
        surface.PrintSummary()

        Dim newPoints = New DataFrame()
        newPoints.AddNumericColumn("x", {0.25, 0.75})
        newPoints.AddNumericColumn("y", {0.5, 0.25})
        Dim pred = surface.Predict(newPoints)
        Console.WriteLine($"Predictions: [{pred(0):F6}, {pred(1):F6}]")
    End Sub
End Module
