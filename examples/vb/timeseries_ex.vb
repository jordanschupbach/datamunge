' CMD: just TARGET=timeseries_ex run-vb
' Time series: ARIMA, seasonal ARIMA and exponential smoothing (SES, Holt, Holt-Winters).
Option Strict On
Imports System
Imports System.Collections.Generic
Imports System.Linq
Imports Datamunge

Module Example
    Private rng As New Random(42)

    Private Sub Seed(n As Integer)
        rng = New Random(n)
    End Sub

    Private Function Uniform() As Double
        Return 1 - rng.NextDouble()  ' (0, 1]: avoids log(0) in the Box-Muller transform.
    End Function

    Private Function Gauss() As Double
        Return Math.Sqrt(-2 * Math.Log(Uniform())) * Math.Cos(2 * Math.PI * Uniform())
    End Function

    Private Function Joined(v As IEnumerable(Of Double)) As String
        Return String.Join(", ", v)
    End Function

    Sub Main()
        Console.WriteLine("=================== ARIMA(1,1,1) on a simulated random walk with drift ===================")
        Seed(42)
        Dim y As New List(Of Double)
        Dim level = 100.0, prevShock = 0.0
        For i = 0 To 149
            Dim shock = Gauss()
            level += 0.3 + 0.4 * prevShock + shock  ' integrated ARMA(0,1) increments with drift
            y.Add(level)
            prevShock = shock
        Next

        Dim model = New ARIMA(y, New ARIMAOptions With {.P = 1UL, .D = 1UL, .Q = 1UL, .DePopulationSize = 80UL, .DeMaxGenerations = 400UL})

        Console.WriteLine($"AR coefficient: {model.ArCoefficients()(0)}")
        Console.WriteLine($"MA coefficient: {model.MaCoefficients()(0)}")
        Console.WriteLine($"sigma^2: {model.Sigma2()}, AIC: {model.Aic()}, BIC: {model.Bic()}")

        Dim forecast = model.ForecastWithIntervals(6UL)
        Console.WriteLine($"6-step forecast: {Joined(forecast.Item1)}")
        Console.WriteLine($"forecast std. errors: {Joined(forecast.Item2)}")

        Console.WriteLine($"{vbLf}=================== SARIMA(1,0,0)(1,1,0)_12 on a seasonal series ===================")
        Seed(7)
        Dim s As New List(Of Double)
        Dim prev = 0.0
        For i = 0 To 119
            prev = 0.5 * prev + Gauss()
            s.Add(20 + 0.2 * i + 5 * Math.Sin(2 * Math.PI * i / 12) + prev)
        Next

        Dim model2 = New ARIMA(s, New ARIMAOptions With {
            .P = 1UL,
            .SeasonalP = 1UL,
            .SeasonalD = 1UL,
            .SeasonalPeriod = 12UL,
            .DePopulationSize = 100UL,
            .DeMaxGenerations = 500UL
        })
        Console.WriteLine($"AR coefficient: {model2.ArCoefficients()(0)}, seasonal AR coefficient: {model2.SeasonalArCoefficients()(0)}")
        Console.WriteLine($"12-step forecast: {Joined(model2.Forecast(12UL))}")

        Console.WriteLine($"{vbLf}=================== Exponential smoothing: Holt-Winters on retail-style seasonal data ===================")
        Dim seasonalShape = {0.8, 0.75, 0.9, 0.95, 1.0, 1.05, 1.1, 1.05, 1.0, 1.1, 1.3, 1.6}
        Seed(11)
        Dim y2 = Enumerable.Range(0, 48).Select(Function(i) (100 + 2 * i) * seasonalShape(i Mod 12) + Gauss() * 3).ToList()

        Dim esModel = New ExponentialSmoothing(y2, New ExponentialSmoothingOptions With {
            .Trend = TrendType.Additive,
            .Seasonal = SeasonalType.Multiplicative,
            .SeasonalPeriod = 12UL,
            .DePopulationSize = 60UL,
            .DeMaxGenerations = 300UL
        })
        Console.WriteLine($"alpha={esModel.Alpha()} beta={esModel.Beta()} gamma={esModel.Gamma()}")
        Console.WriteLine($"sigma^2: {esModel.Sigma2()}, AIC: {esModel.Aic()}")
        Console.WriteLine($"12-month forecast: {Joined(esModel.Forecast(12UL))}")

        Console.WriteLine($"{vbLf}=================== Simple exponential smoothing vs. Holt's linear trend ===================")
        Dim flat = {50.2, 49.8, 50.5, 49.6, 50.1, 50.3, 49.9, 50.0, 50.4, 49.7}

        Dim ses = New ExponentialSmoothing(flat, New ExponentialSmoothingOptions())
        Console.WriteLine($"SES alpha= {ses.Alpha()}")
        Console.WriteLine($"SES 5-step forecast: {Joined(ses.Forecast(5UL))}")

        Dim holt = New ExponentialSmoothing(flat, New ExponentialSmoothingOptions With {.Trend = TrendType.Additive})
        Console.WriteLine($"Holt alpha={holt.Alpha()} beta={holt.Beta()}")
        Console.WriteLine($"Holt 5-step forecast: {Joined(holt.Forecast(5UL))}")
    End Sub
End Module
