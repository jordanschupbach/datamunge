// CMD: just TARGET=autodiff_ex run-go

//go:build ignore

// Automatic differentiation: forward mode (Dual), reverse mode (Tape/Var), Hessians (HyperDual).
package main

import (
	"datamunge"
	"fmt"
	"math"
	"strings"
)

// Note: the C++ autodiff_ex.cpp additionally demonstrates the generic template drivers
// (derivative/gradient_forward/gradient_reverse/jacobian_forward/hessian), which can't be
// bound. This port demonstrates the same ideas with the bound named-method API.
// v.Index() is the Var's position in the adjoint slice tape.Backward(output) returns.

func dual(value, derivative float64) *datamunge.Dual {
	return datamunge.Must(datamunge.NewDualWithDerivative(value, derivative))
}

func rosenbrockDual(x0, x1 *datamunge.Dual) *datamunge.Dual {
	a := datamunge.Must(dual(1.0, 0.0).Subtract(x0))
	b := datamunge.Must(x1.Subtract(datamunge.Must(x0.Multiply(x0))))
	bb := datamunge.Must(datamunge.Must(b.Multiply(b)).MultiplyScalar(100.0))
	return datamunge.Must(datamunge.Must(a.Multiply(a)).Add(bb))
}

func rosenbrockValue(x0, x1 float64) float64 {
	return math.Pow(1.0-x0, 2) + 100.0*math.Pow(x1-x0*x0, 2)
}

// Build the Rosenbrock graph on tape; returns the output Var.
func rosenbrockVar(tape *datamunge.Tape, x0, x1 *datamunge.Var) *datamunge.Var {
	a := datamunge.Must(datamunge.Must(datamunge.NewVar(tape, 1.0)).Subtract(x0))
	b := datamunge.Must(x1.Subtract(datamunge.Must(x0.Multiply(x0))))
	bb := datamunge.Must(datamunge.Must(b.Multiply(b)).MultiplyScalar(100.0))
	return datamunge.Must(datamunge.Must(a.Multiply(a)).Add(bb))
}

func rosenbrockHyperDual(x0, x1 *datamunge.HyperDual) *datamunge.HyperDual {
	one := datamunge.Must(datamunge.NewHyperDualWithEps1eps2(1.0, 0.0, 0.0, 0.0))
	a := datamunge.Must(one.Subtract(x0))
	b := datamunge.Must(x1.Subtract(datamunge.Must(x0.Multiply(x0))))
	bb := datamunge.Must(datamunge.Must(b.Multiply(b)).MultiplyScalar(100.0))
	return datamunge.Must(datamunge.Must(a.Multiply(a)).Add(bb))
}

func adjoint(adjoints []float64, v *datamunge.Var) float64 {
	return adjoints[datamunge.Must(v.Index())]
}

func vectorFn(x, y *datamunge.Dual) []*datamunge.Dual {
	return []*datamunge.Dual{
		datamunge.Must(x.Multiply(x)),
		datamunge.Must(x.Multiply(y)),
		datamunge.Must(datamunge.Must(y.Multiply(y)).Multiply(y)),
	}
}

func join(values []float64) string {
	parts := make([]string, len(values))
	for i, v := range values {
		parts[i] = fmt.Sprint(v)
	}
	return "[" + strings.Join(parts, ", ") + "]"
}

func main() {
	fmt.Println("=================== Forward mode: scalar derivative ===================")
	x0 := dual(2.0, 1.0) // seed derivative = 1 to read df/dx directly
	cube := datamunge.Must(datamunge.Must(x0.Multiply(x0)).Multiply(x0))
	f := datamunge.Must(cube.Subtract(datamunge.Must(x0.MultiplyScalar(2.0)))) // x^3 - 2x
	fmt.Println("f(x) = x^3 - 2x, f'(2) =", datamunge.Must(f.Derivative()), "(exact: 10)")

	fmt.Println("\n=================== Reverse mode: build a graph by hand ===================")
	tape := datamunge.Must(datamunge.NewTape())
	a := datamunge.Must(datamunge.NewVar(tape, 2.0))
	b := datamunge.Must(datamunge.NewVar(tape, 3.0))
	y := datamunge.Must(datamunge.Must(a.Multiply(b)).Add(datamunge.Must(a.Sin())))
	fmt.Println("y = a*b + sin(a) at a=2, b=3 -> y =", datamunge.Must(y.Value()))
	adj := datamunge.Must(tape.Backward(y))
	fmt.Println("dy/da =", adjoint(adj, a), "(exact: b + cos(a))")
	fmt.Println("dy/db =", adjoint(adj, b), "(exact: a)")

	fmt.Println("\n=================== Forward vs reverse mode agree on the Rosenbrock function ===================")
	px, py := 0.0, 0.0
	// Forward mode: one Dual pass per partial derivative, seeding the direction of interest.
	gx := datamunge.Must(rosenbrockDual(dual(px, 1.0), dual(py, 0.0)).Derivative())
	gy := datamunge.Must(rosenbrockDual(dual(px, 0.0), dual(py, 1.0)).Derivative())
	// Reverse mode: one Tape/Var pass computes every partial at once.
	tape = datamunge.Must(datamunge.NewTape())
	vx := datamunge.Must(datamunge.NewVar(tape, px))
	vy := datamunge.Must(datamunge.NewVar(tape, py))
	grad := datamunge.Must(tape.Backward(rosenbrockVar(tape, vx, vy)))
	fmt.Println("f(0,0) =", rosenbrockValue(px, py))
	fmt.Println("gradient (forward mode):", join([]float64{gx, gy}))
	fmt.Println("gradient (reverse mode):", join([]float64{adjoint(grad, vx), adjoint(grad, vy)}))

	fmt.Println("\n=================== Jacobian of a vector-valued function ===================")
	// f(x,y) = [x^2, xy, y^3] at (2,3) -- one Dual pass per (output, input) pair.
	jac := make([][]float64, 3)
	for i := range jac {
		jac[i] = make([]float64, 2)
	}
	seeds := [][]float64{{1.0, 0.0}, {0.0, 1.0}}
	for col := 0; col < 2; col++ {
		outs := vectorFn(dual(2.0, seeds[col][0]), dual(3.0, seeds[col][1]))
		for row, out := range outs {
			jac[row][col] = datamunge.Must(out.Derivative())
		}
	}
	fmt.Println("f(x,y) = [x^2, xy, y^3] at (2,3), Jacobian:")
	for _, r := range jac {
		fmt.Println("  " + join(r))
	}

	fmt.Println("\n=================== Hessian via second-order forward mode (HyperDual) ===================")
	// One HyperDual pass per (i, j) pair reads off d^2f/dxi dxj directly from eps1eps2.
	mx, my := 1.0, 1.0 // the Rosenbrock function's minimum
	h := [][]float64{{0, 0}, {0, 0}}
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			e1, e2 := seeds[i], seeds[j]
			hx := datamunge.Must(datamunge.NewHyperDualWithEps1eps2(mx, e1[0], e2[0], 0.0))
			hy := datamunge.Must(datamunge.NewHyperDualWithEps1eps2(my, e1[1], e2[1], 0.0))
			h[i][j] = datamunge.Must(rosenbrockHyperDual(hx, hy).Eps1eps2())
		}
	}
	fmt.Println("Hessian of the Rosenbrock function at its minimum (1,1):")
	for _, r := range h {
		fmt.Println("  " + join(r))
	}

	fmt.Println("\n=================== Gradient descent driven by reverse-mode gradients ===================")
	point := []float64{-1.2, 1.0} // the classic Rosenbrock start
	learningRate := 0.001
	nSteps := 2000
	for step := 0; step < nSteps; step++ {
		func() {
			stepTape := datamunge.Must(datamunge.NewTape())
			defer stepTape.Close()
			sx := datamunge.Must(datamunge.NewVar(stepTape, point[0]))
			sy := datamunge.Must(datamunge.NewVar(stepTape, point[1]))
			vf := rosenbrockVar(stepTape, sx, sy)
			g := datamunge.Must(stepTape.Backward(vf))
			loss := datamunge.Must(vf.Value())
			point[0] -= learningRate * adjoint(g, sx)
			point[1] -= learningRate * adjoint(g, sy)
			if step == 0 || step == nSteps-1 {
				fmt.Printf("step %d: loss = %v, x = %s\n", step, loss, join(point))
			}
		}()
	}
	fmt.Println("(true minimum is at [1, 1] with loss 0)")
}
