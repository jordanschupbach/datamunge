' CMD: just TARGET=optim_advanced_ex run-vb
' Advanced datamunge optimizers (ProximalGradient, FISTA, LevenbergMarquardt, Newton, TrustRegionNewton,
' AugmentedLagrangian, SQP, InteriorPoint, BayesianOptimization, RBFGaussianProcessSurrogate) from Visual Basic.
'
' Every one of these optimizers takes an interface (ProximalFunction / HessianFunction /
' EqualityConstrainedFunction / InequalityConstrainedFunction / ResidualFunction, or
' ArbitraryFunction for BayesianOptimization) that the C++ library leaves abstract: callers
' implement the objective. From Visual Basic, inherit from the interface class and override its
' overridable methods; the optimizer then calls the Visual Basic overrides. Coordinates are updated
' in place through a ByRef parameter (the C++ parameter is a non-const std::vector<double>&).
Option Strict On
Imports System
Imports System.Collections.Generic
Imports System.Linq
Imports Datamunge

Module Example
    Sub Main()
        Console.WriteLine("=================== RBFGaussianProcessSurrogate (concrete BayesianSurrogate) ===================")
        ' Fit a 1D surrogate to a simple bowl y = (x-1)^2 sampled at a few points,
        ' then check the acquisition function favors an unexplored region.
        Dim xs = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0}
        Dim ys = xs.Select(Function(x) (x - 1) * (x - 1)).ToArray()
        Dim surrogate = New RBFGaussianProcessSurrogate(1.0, 0.000001)
        surrogate.Fit(xs.Select(Function(x) New Double() {x}), ys)
        Dim incumbent = Math.Min(0.0, ys.Min())  ' best observed value so far
        Console.WriteLine($"incumbent (best observed y): {incumbent}")
        For Each x In {0.25, 1.0, 2.5}
            Console.WriteLine($"acquisition(x={x:F2}) = {surrogate.Acquisition({x}, incumbent):F6}")
        Next

        Console.WriteLine($"{vbLf}=================== Options structs + optimizer construction (all 10 new classes) ===================")
        ' Every new Options struct and optimizer class constructs cleanly, proving
        ' the binding for the whole batch is wired correctly.
        Dim pg = New ProximalGradient(New ProximalGradientOptions With {.StepSize = 0.05})
        Console.WriteLine($"ProximalGradient: {pg}")
        Dim fista = New FISTA(New FISTAOptions())
        Console.WriteLine($"FISTA: {fista}")
        Dim lm = New LevenbergMarquardt(New LevenbergMarquardtOptions With {.MaxIterations = 200UL})
        Console.WriteLine($"LevenbergMarquardt: {lm}")
        Dim newton = New Newton(New NewtonOptions())
        Console.WriteLine($"Newton: {newton}")
        Dim trn = New TrustRegionNewton(New TrustRegionNewtonOptions())
        Console.WriteLine($"TrustRegionNewton: {trn}")
        Dim al = New AugmentedLagrangian(New AugmentedLagrangianOptions())
        Console.WriteLine($"AugmentedLagrangian: {al}")
        Dim sqp = New SQP(New SQPOptions())
        Console.WriteLine($"SQP: {sqp}")
        Dim ip = New InteriorPoint(New InteriorPointOptions())
        Console.WriteLine($"InteriorPoint: {ip}")
        Dim bo = New BayesianOptimization(New BayesianOptimizationOptions With {.InitialSamples = 8UL})
        Console.WriteLine($"BayesianOptimization: {bo}")

        Console.WriteLine($"{vbLf}=================== Visual Basic objectives (subclassing the C++ interfaces) ===================")
        ' Newton's method on f(x, y) = (x - 1)^2 + 10 (y + 2)^2 with exact gradient and Hessian.
        Dim start As New List(Of Double) From {5.0, 5.0}
        Dim value = newton.Optimize(New Quadratic(), start)
        Console.WriteLine($"Newton: f = {value:G6} at ({start(0):F6}, {start(1):F6}) (exact minimum 0 at (1, -2))")

        ' Levenberg-Marquardt fitting y = a * exp(b t) to exact data from a = 2, b = 0.5.
        Dim parameters As New List(Of Double) From {1.0, 0.1}
        Dim sse = lm.Optimize(New Exponential({0.0, 0.5, 1.0, 1.5, 2.0}), parameters)
        Console.WriteLine($"LevenbergMarquardt: residual = {sse:G3}, a = {parameters(0):F6}, b = {parameters(1):F6} (exact: 2, 0.5)")

        ' An interface method a subclass leaves out throws when the optimizer calls it.
        Dim err As Exception = Nothing
        Try
            Dim origin As New List(Of Double) From {0.0, 0.0}
            newton.Optimize(New Incomplete(), origin)
        Catch e As Exception
            err = e
        End Try
        Console.WriteLine($"missing gradient(): ok={err Is Nothing}, error mentions it: {err IsNot Nothing AndAlso err.Message.Contains("gradient")}")
    End Sub
End Module

Class Quadratic
    Inherits HessianFunction

    Public Overrides Function Evaluate(coordinates As IEnumerable(Of Double)) As Double
        Dim c = coordinates.ToList()
        Return Math.Pow(c(0) - 1, 2) + 10 * Math.Pow(c(1) + 2, 2)
    End Function

    Public Overrides Function Gradient(coordinates As IEnumerable(Of Double)) As List(Of Double)
        Dim c = coordinates.ToList()
        Return New List(Of Double) From {2 * (c(0) - 1), 20 * (c(1) + 2)}
    End Function

    Public Overrides Function Hessian(coordinates As IEnumerable(Of Double)) As List(Of List(Of Double))
        Return New List(Of List(Of Double)) From {New List(Of Double) From {2.0, 0.0}, New List(Of Double) From {0.0, 20.0}}
    End Function
End Class

Class Exponential
    Inherits ResidualFunction

    Private ReadOnly ts As Double()

    Public Sub New(ts As Double())
        Me.ts = ts
    End Sub

    Public Overrides Function Residuals(coordinates As IEnumerable(Of Double)) As List(Of Double)
        Dim p = coordinates.ToList()
        Return ts.Select(Function(t) p(0) * Math.Exp(p(1) * t) - 2 * Math.Exp(0.5 * t)).ToList()
    End Function

    Public Overrides Function Jacobian(coordinates As IEnumerable(Of Double)) As List(Of List(Of Double))
        Dim p = coordinates.ToList()
        Return ts.Select(Function(t) New List(Of Double) From {Math.Exp(p(1) * t), p(0) * t * Math.Exp(p(1) * t)}).ToList()
    End Function
End Class

Class Incomplete
    Inherits HessianFunction

    Public Overrides Function Evaluate(coordinates As IEnumerable(Of Double)) As Double
        Return 0.0
    End Function
End Class
