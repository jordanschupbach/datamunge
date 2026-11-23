' CMD: just TARGET=autodiff_ex run-vb
' Automatic differentiation: forward mode (Dual), reverse mode (Tape/Var), Hessians (HyperDual).
Option Strict On
Imports System
Imports System.Collections.Generic
Imports Datamunge

' Note: the C++ autodiff_ex.cpp additionally demonstrates the generic template drivers
' (derivative/gradient_forward/gradient_reverse/jacobian_forward/hessian), which can't be
' bound. This port demonstrates the same ideas with the bound named-method API.
' Var.Index() is the Var's position in the adjoint vector Tape.Backward returns.
Module Example
    Private Function RosenbrockDual(x0 As Dual, x1 As Dual) As Dual
        Dim a = New Dual(1.0, 0.0).Subtract(x0)
        Dim b = x1.Subtract(x0.Multiply(x0))
        Return a.Multiply(a).Add(b.Multiply(b).MultiplyScalar(100.0))
    End Function

    Private Function RosenbrockValue(x0 As Double, x1 As Double) As Double
        Return Math.Pow(1.0 - x0, 2) + 100.0 * Math.Pow(x1 - x0 * x0, 2)
    End Function

    ' Build the Rosenbrock graph on tape; returns the output Var.
    Private Function RosenbrockVar(tape As Tape, x0 As Var, x1 As Var) As Var
        Dim a = New Var(tape, 1.0).Subtract(x0)
        Dim b = x1.Subtract(x0.Multiply(x0))
        Return a.Multiply(a).Add(b.Multiply(b).MultiplyScalar(100.0))
    End Function

    Private Function RosenbrockHyperDual(x0 As HyperDual, x1 As HyperDual) As HyperDual
        Dim a = New HyperDual(1.0, 0.0, 0.0, 0.0).Subtract(x0)
        Dim b = x1.Subtract(x0.Multiply(x0))
        Return a.Multiply(a).Add(b.Multiply(b).MultiplyScalar(100.0))
    End Function

    Private Function Adjoint(adjoints As List(Of Double), v As Var) As Double
        Return adjoints(CInt(v.Index()))
    End Function

    ' f(x,y) = [x^2, xy, y^3]
    Private Function VectorFn(x As Dual, y As Dual) As Dual()
        Return {x.Multiply(x), x.Multiply(y), y.Multiply(y).Multiply(y)}
    End Function

    Sub Main()
        Console.WriteLine("=================== Forward mode: scalar derivative ===================")
        Block1()
        Console.WriteLine($"{vbLf}=================== Reverse mode: build a graph by hand ===================")
        Block2()
        Console.WriteLine($"{vbLf}=================== Forward vs reverse mode agree on the Rosenbrock function ===================")
        Block3()
        Console.WriteLine($"{vbLf}=================== Jacobian of a vector-valued function ===================")
        Block4()
        Console.WriteLine($"{vbLf}=================== Hessian via second-order forward mode (HyperDual) ===================")
        Block5()
        Console.WriteLine($"{vbLf}=================== Gradient descent driven by reverse-mode gradients ===================")
        Block6()
    End Sub

    Private Sub Block1()
        Dim x0 = New Dual(2.0, 1.0)  ' seed derivative = 1 to read df/dx directly
        Dim f = x0.Multiply(x0).Multiply(x0).Subtract(x0.MultiplyScalar(2.0))  ' x^3 - 2x
        Console.WriteLine($"f(x) = x^3 - 2x, f'(2) = {f.Derivative()} (exact: 10)")
    End Sub

    Private Sub Block2()
        Dim tape = New Tape()
        Dim a = New Var(tape, 2.0)
        Dim b = New Var(tape, 3.0)
        Dim y = a.Multiply(b).Add(a.Sin())
        Console.WriteLine($"y = a*b + sin(a) at a=2, b=3 -> y = {y.Value()}")
        Dim adj = tape.Backward(y)
        Console.WriteLine($"dy/da = {Adjoint(adj, a)} (exact: b + cos(a))")
        Console.WriteLine($"dy/db = {Adjoint(adj, b)} (exact: a)")
    End Sub

    Private Sub Block3()
        Dim px = 0.0, py = 0.0
        ' Forward mode: one Dual pass per partial derivative, seeding the direction of interest.
        Dim gx = RosenbrockDual(New Dual(px, 1.0), New Dual(py, 0.0)).Derivative()
        Dim gy = RosenbrockDual(New Dual(px, 0.0), New Dual(py, 1.0)).Derivative()
        ' Reverse mode: one Tape/Var pass computes every partial at once.
        Dim tape = New Tape()
        Dim vx = New Var(tape, px)
        Dim vy = New Var(tape, py)
        Dim grad = tape.Backward(RosenbrockVar(tape, vx, vy))
        Console.WriteLine($"f(0,0) = {RosenbrockValue(px, py)}")
        Console.WriteLine($"gradient (forward mode): [{gx}, {gy}]")
        Console.WriteLine($"gradient (reverse mode): [{Adjoint(grad, vx)}, {Adjoint(grad, vy)}]")
    End Sub

    ' f(x,y) = [x^2, xy, y^3] at (2,3) -- one Dual pass per (output, input) pair.
    Private Sub Block4()
        Dim jac(2, 1) As Double
        For Each seed In {(Col:=0, Sx:=1.0, Sy:=0.0), (Col:=1, Sx:=0.0, Sy:=1.0)}
            Dim outputs = VectorFn(New Dual(2.0, seed.Sx), New Dual(3.0, seed.Sy))
            For row = 0 To outputs.Length - 1
                jac(row, seed.Col) = outputs(row).Derivative()
            Next
        Next
        Console.WriteLine("f(x,y) = [x^2, xy, y^3] at (2,3), Jacobian:")
        For row = 0 To 2
            Console.WriteLine($"  [{jac(row, 0)}, {jac(row, 1)}]")
        Next
    End Sub

    ' One HyperDual pass per (i, j) pair reads off d^2f/dxi dxj directly from eps1eps2.
    Private Sub Block5()
        Dim mx = 1.0, my = 1.0  ' the Rosenbrock function's minimum
        Dim seeds As (X As Double, Y As Double)() = {(1.0, 0.0), (0.0, 1.0)}
        Dim h(1, 1) As Double
        For i = 0 To 1
            For j = 0 To 1
                Dim e1 = seeds(i), e2 = seeds(j)
                h(i, j) = RosenbrockHyperDual(New HyperDual(mx, e1.X, e2.X, 0.0), New HyperDual(my, e1.Y, e2.Y, 0.0)).Eps1eps2()
            Next
        Next
        Console.WriteLine("Hessian of the Rosenbrock function at its minimum (1,1):")
        For i = 0 To 1
            Console.WriteLine($"  [{h(i, 0)}, {h(i, 1)}]")
        Next
    End Sub

    Private Sub Block6()
        Dim point = {-1.2, 1.0}  ' the classic Rosenbrock start
        Const learningRate As Double = 0.001
        Const nSteps As Integer = 2000
        For stepIndex = 0 To nSteps - 1
            Dim tape = New Tape()
            Dim vx = New Var(tape, point(0))
            Dim vy = New Var(tape, point(1))
            Dim vf = RosenbrockVar(tape, vx, vy)
            Dim grad = tape.Backward(vf)
            Dim loss = vf.Value()
            point(0) -= learningRate * Adjoint(grad, vx)
            point(1) -= learningRate * Adjoint(grad, vy)
            If stepIndex = 0 OrElse stepIndex = nSteps - 1 Then
                Console.WriteLine($"step {stepIndex}: loss = {loss}, x = [{point(0)}, {point(1)}]")
            End If
        Next
        Console.WriteLine("(true minimum is at [1, 1] with loss 0)")
    End Sub
End Module
