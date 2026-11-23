' CMD: just TARGET=ode_ex run-vb
' ODEs: built-in named systems, harmonic-oscillator energy check and a Lorenz phase plot.
Option Strict On
Imports System
Imports System.Collections.Generic
Imports System.Linq
Imports Datamunge

Module Example
    ' Approximate C's %g: six significant digits, no padding.
    Private Function FmtG(x As Double) As String
        Return x.ToString("G6")
    End Function

    Private Function LastState(sol As ODESolution) As List(Of Double)
        Return sol.StateAt(sol.Size() - 1UL)
    End Function

    ' The time vector followed by the requested state component (0-based), one value per step.
    Private Function SolutionSeries(sol As ODESolution, ParamArray components As Integer()) As (Times As List(Of Double), Components As List(Of Double)())
        Dim steps = Enumerable.Range(0, CInt(sol.Size())).Select(Function(i) CULng(i)).ToList()
        Dim states = steps.Select(Function(i) sol.StateAt(i)).ToList()
        Return (steps.Select(Function(i) sol.TimeAt(i)).ToList(), components.Select(Function(c) states.Select(Function(s) s(c)).ToList()).ToArray())
    End Function

    Private Function Rk4Solver(stepSize As Double) As ODESolver
        Return New ODESolver(New ODEOptions With {.Method = StepMethod.RK4, .StepSize = stepSize})
    End Function

    ' NOTE: like the Lua version this example uses the built-in named systems via SolveBuiltin.

    Sub Main()
        Console.WriteLine("=================== Every named built-in system (no director needed) ===================")
        Dim solver = New ODESolver()
        Dim systems As (Name As String, Params As Double(), Y0 As Double())() = {
            ("exponential_decay", {1.0}, {1.0}),
            ("logistic_growth", {1.0, 1.0}, {0.5}),
            ("harmonic_oscillator", {1.0}, {1.0, 0.0}),
            ("van_der_pol", {1.0}, {2.0, 0.0}),
            ("lorenz", {10.0, 28.0, 8.0 / 3}, {1.0, 1.0, 1.0})
        }
        For Each odeSystem In systems
            Dim sol = solver.SolveBuiltin(odeSystem.Name, odeSystem.Params, odeSystem.Y0, 0.0, 1.0)
            Console.WriteLine($"{odeSystem.Name,-22} steps={sol.StepsTaken,-6} final state=[{String.Join(", ", LastState(sol).Select(AddressOf FmtG))}]")
        Next

        Console.WriteLine($"{vbLf}=================== Harmonic oscillator (energy conservation) ===================")
        HarmonicOscillator()

        Console.WriteLine($"{vbLf}=================== Lorenz attractor (phase plane) ===================")
        Lorenz()
    End Sub

    Private Sub HarmonicOscillator()
        Dim sol = Rk4Solver(0.01).SolveBuiltin("harmonic_oscillator", {1.0}, {1.0, 0.0}, 0.0, 20.0)
        Dim last = LastState(sol)
        Dim x = last(0), v = last(1)
        Console.WriteLine($"x(20) = {x:F6} (cos(20) = {Math.Cos(20.0):F6})")
        Console.WriteLine($"energy x^2+v^2 = {x * x + v * v:F6} (should stay near 1.0)")
        Dim series = SolutionSeries(sol, 0, 1)
        Dim ts = series.Times
        Dim p = RPlot.Plot(ts, series.Components(0), "l", "x(t)")
        p.Lines(ts, series.Components(1), "v(t)")
        p.Title("Harmonic Oscillator").XLabel("t").YLabel("state")
        p.SaveSvg("ode_harmonic_oscillator_vb.svg")
        Console.WriteLine("wrote ode_harmonic_oscillator_vb.svg")
    End Sub

    Private Sub Lorenz()
        Dim sol = Rk4Solver(0.005).SolveBuiltin("lorenz", {10.0, 28.0, 8.0 / 3}, {1.0, 1.0, 1.0}, 0.0, 25.0)
        Console.WriteLine($"steps_taken = {sol.StepsTaken}")
        Dim series = SolutionSeries(sol, 0, 2).Components
        Dim p = RPlot.Plot(series(0), series(1), "l", "trajectory")
        p.Title("Lorenz Attractor (x-z phase plane)").XLabel("x").YLabel("z")
        p.SaveSvg("ode_lorenz_phase_plane_vb.svg")
        Console.WriteLine("wrote ode_lorenz_phase_plane_vb.svg")
    End Sub
End Module
