package main

import (
	"datamunge"
	"fmt"
)

func rosenbrockDual(x0, x1 datamunge.Dual) datamunge.Dual {
	a := datamunge.NewDual(1.0, 0.0).Subtract(x0)
	b := x1.Subtract(x0.Multiply(x0))
	return a.Multiply(a).Add(b.Multiply(b).Multiply_scalar(100.0))
}

func rosenbrockValue(x0, x1 float64) float64 {
	a := 1.0 - x0
	b := x1 - x0*x0
	return a*a + 100.0*b*b
}

func rosenbrockHyperdual(x0, x1 datamunge.HyperDual) datamunge.HyperDual {
	a := datamunge.NewHyperDual(1.0, 0.0, 0.0, 0.0).Subtract(x0)
	b := x1.Subtract(x0.Multiply(x0))
	return a.Multiply(a).Add(b.Multiply(b).Multiply_scalar(100.0))
}

func main() {
	fmt.Println("=================== Forward mode: scalar derivative ===================")
	x0 := datamunge.NewDual(2.0, 1.0)
	f := x0.Multiply(x0).Multiply(x0).Subtract(x0.Multiply_scalar(2.0))
	fmt.Printf("f(x) = x^3 - 2x, f'(2) = %g (exact: 10)\n", f.Derivative())

	fmt.Println("\n=================== Reverse mode: build a graph by hand ===================")
	tape := datamunge.NewTape()
	a := datamunge.NewVar(tape, 2.0)
	b := datamunge.NewVar(tape, 3.0)
	y := a.Multiply(b).Add(a.Sin())
	fmt.Printf("y = a*b + sin(a) at a=2, b=3 -> y = %g\n", y.Value())
	adjoint := tape.Backward(y)
	fmt.Printf("dy/da = %g (exact: b + cos(a))\n", adjoint.Get(int(a.Index())))
	fmt.Printf("dy/db = %g (exact: a)\n", adjoint.Get(int(b.Index())))

	fmt.Println("\n=================== Forward vs reverse mode agree on the Rosenbrock function ===================")
	px := 0.0
	py := 0.0

	gx := rosenbrockDual(datamunge.NewDual(px, 1.0), datamunge.NewDual(py, 0.0)).Derivative()
	gy := rosenbrockDual(datamunge.NewDual(px, 0.0), datamunge.NewDual(py, 1.0)).Derivative()

	tape2 := datamunge.NewTape()
	vx := datamunge.NewVar(tape2, px)
	vy := datamunge.NewVar(tape2, py)
	va := datamunge.NewVar(tape2, 1.0).Subtract(vx)
	vb := vy.Subtract(vx.Multiply(vx))
	vf := va.Multiply(va).Add(vb.Multiply(vb).Multiply_scalar(100.0))
	gradRev := tape2.Backward(vf)

	fmt.Printf("f(0,0) = %g\n", rosenbrockValue(px, py))
	fmt.Printf("gradient (forward mode): [%g, %g]\n", gx, gy)
	fmt.Printf("gradient (reverse mode): [%g, %g]\n", gradRev.Get(int(vx.Index())), gradRev.Get(int(vy.Index())))

	fmt.Println("\n=================== Jacobian of a vector-valued function ===================")
	vxVal := 2.0
	vyVal := 3.0

	var jac [3][2]float64
	cols := [][3]float64{{0, 1.0, 0.0}, {1, 0.0, 1.0}}
	for _, c := range cols {
		col := int(c[0])
		seedX := c[1]
		seedY := c[2]
		ox := datamunge.NewDual(vxVal, seedX).Multiply(datamunge.NewDual(vxVal, seedX))
		oy := datamunge.NewDual(vxVal, seedX).Multiply(datamunge.NewDual(vyVal, seedY))
		oz := datamunge.NewDual(vyVal, seedY).Multiply(datamunge.NewDual(vyVal, seedY)).Multiply(datamunge.NewDual(vyVal, seedY))
		jac[0][col] = ox.Derivative()
		jac[1][col] = oy.Derivative()
		jac[2][col] = oz.Derivative()
	}
	fmt.Println("f(x,y) = [x^2, xy, y^3] at (2,3), Jacobian:")
	for row := 0; row < 3; row++ {
		fmt.Printf("  [%g, %g]\n", jac[row][0], jac[row][1])
	}

	fmt.Println("\n=================== Hessian via second-order forward mode (HyperDual) ===================")
	mx := 1.0
	my := 1.0

	seeds := [2][2]float64{{1.0, 0.0}, {0.0, 1.0}}
	var h [2][2]float64
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			e1 := seeds[i]
			e2 := seeds[j]
			hx := datamunge.NewHyperDual(mx, e1[0], e2[0], 0.0)
			hy := datamunge.NewHyperDual(my, e1[1], e2[1], 0.0)
			h[i][j] = rosenbrockHyperdual(hx, hy).Eps1eps2()
		}
	}
	fmt.Println("Hessian of the Rosenbrock function at its minimum (1,1):")
	for row := 0; row < 2; row++ {
		fmt.Printf("  [%g, %g]\n", h[row][0], h[row][1])
	}

	fmt.Println("\n=================== Gradient descent driven by reverse-mode gradients ===================")
	pointX := -1.2
	pointY := 1.0
	learningRate := 0.001
	nSteps := 2000
	for step := 0; step < nSteps; step++ {
		t := datamunge.NewTape()
		vx2 := datamunge.NewVar(t, pointX)
		vy2 := datamunge.NewVar(t, pointY)
		va2 := datamunge.NewVar(t, 1.0).Subtract(vx2)
		vb2 := vy2.Subtract(vx2.Multiply(vx2))
		vf2 := va2.Multiply(va2).Add(vb2.Multiply(vb2).Multiply_scalar(100.0))
		grad := t.Backward(vf2)
		loss := vf2.Value()
		pointX -= learningRate * grad.Get(int(vx2.Index()))
		pointY -= learningRate * grad.Get(int(vy2.Index()))
		if step == 0 || step == nSteps-1 {
			fmt.Printf("step %d: loss = %g, x = [%g, %g]\n", step, loss, pointX, pointY)
		}
	}
	fmt.Println("(true minimum is at [1, 1] with loss 0)")
}
