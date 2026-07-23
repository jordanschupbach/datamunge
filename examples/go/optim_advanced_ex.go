package main

import (
	"datamunge"
	"fmt"
	"math"
)

// ---- Go-side helpers for building/reading the SWIG vector types. ----

func dvector(values []float64) datamunge.DVector {
	out := datamunge.NewDVector(int64(len(values)))
	for i, v := range values {
		out.Set(i, v)
	}
	return out
}

func dvectorSlice(v datamunge.DVector) []float64 {
	n := int(v.Size())
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		out[i] = v.Get(i)
	}
	return out
}

func dvectorvector(rows [][]float64) datamunge.DVectorVector {
	out := datamunge.NewDVectorVector(int64(len(rows)))
	for i, row := range rows {
		out.Set(i, dvector(row))
	}
	return out
}

// ---- New function-type interfaces (director-based extension points): subclassing them in
// Go means defining a plain Go struct with matching methods and wrapping it via the
// NewDirectorXxx(v interface{}) constructor SWIG generates for each director-enabled type.
// This is the load-bearing capability under test: does a C++ optimizer, calling back through
// the director shim into this Go struct's methods, actually reach Go code? ----

// QuadraticBowlHessian: f(x) = (x-target)^T A (x-target), a well-conditioned 2-D bowl with a
// known minimum, exposing an exact constant Hessian -- the natural example for Newton's method.
type QuadraticBowlHessian struct {
	target [2]float64
	a      [2][2]float64
}

func newQuadraticBowlHessian(target [2]float64) *QuadraticBowlHessian {
	return &QuadraticBowlHessian{target: target, a: [2][2]float64{{4.0, 0.5}, {0.5, 2.0}}}
}

func (q *QuadraticBowlHessian) Evaluate(coords datamunge.DVector) float64 {
	x := dvectorSlice(coords)
	d0, d1 := x[0]-q.target[0], x[1]-q.target[1]
	return d0*(q.a[0][0]*d0+q.a[0][1]*d1) + d1*(q.a[1][0]*d0+q.a[1][1]*d1)
}

func (q *QuadraticBowlHessian) Gradient(coords datamunge.DVector) datamunge.DVector {
	x := dvectorSlice(coords)
	d0, d1 := x[0]-q.target[0], x[1]-q.target[1]
	return dvector([]float64{
		2.0 * (q.a[0][0]*d0 + q.a[0][1]*d1),
		2.0 * (q.a[1][0]*d0 + q.a[1][1]*d1),
	})
}

func (q *QuadraticBowlHessian) Hessian(coords datamunge.DVector) datamunge.DVectorVector {
	return dvectorvector([][]float64{
		{2.0 * q.a[0][0], 2.0 * q.a[0][1]},
		{2.0 * q.a[1][0], 2.0 * q.a[1][1]},
	})
}

// ExpDecayResidual: residuals for fitting A*exp(-k*t)+c to synthetic noise-free data -- the
// textbook nonlinear least-squares curve fit, and the use case LevenbergMarquardt targets.
type ExpDecayResidual struct {
	ts []float64
	ys []float64
}

func (e *ExpDecayResidual) Residuals(params datamunge.DVector) datamunge.DVector {
	p := dvectorSlice(params)
	a, k, c := p[0], p[1], p[2]
	out := make([]float64, len(e.ts))
	for i, t := range e.ts {
		out[i] = a*math.Exp(-k*t) + c - e.ys[i]
	}
	return dvector(out)
}

func (e *ExpDecayResidual) Jacobian(params datamunge.DVector) datamunge.DVectorVector {
	p := dvectorSlice(params)
	a, k, _ := p[0], p[1], p[2]
	rows := make([][]float64, len(e.ts))
	for i, t := range e.ts {
		expo := math.Exp(-k * t)
		rows[i] = []float64{expo, -a * t * expo, 1.0}
	}
	return dvectorvector(rows)
}

// Shifted1DBowl: a 1-D bowl with a known minimum near x=1.7 and no gradient exposed -- the
// black-box setting BayesianOptimization targets. Only needs ArbitraryFunction (no subclassing
// of BayesianSurrogate is required here -- RBFGaussianProcessSurrogate is ready-to-use).
type Shifted1DBowl struct{}

func (Shifted1DBowl) Evaluate(coords datamunge.DVector) float64 {
	x := dvectorSlice(coords)[0]
	return (x-1.7)*(x-1.7) + 0.1*math.Sin(10.0*x)
}

func main() {
	fmt.Println("=================== HessianFunction: Newton's method on a quadratic bowl ===================")
	target := [2]float64{3.0, -1.5}
	newtonFn := datamunge.NewDirectorHessianFunction(newQuadraticBowlHessian(target))
	x := dvector([]float64{0.0, 0.0})
	newtonOpts := datamunge.NewNewtonOptions()
	newtonOpts.SetMax_iterations(50)
	newtonOpts.SetTolerance(1e-10)
	value := datamunge.NewNewton(newtonOpts).Optimize(newtonFn, x)
	fmt.Printf("Newton:            f=%v x=%v (true minimum: f=0 at %v)\n", value, dvectorSlice(x), target)

	fmt.Println("\n=================== HessianFunction: TrustRegionNewton on the same bowl ===================")
	trnFn := datamunge.NewDirectorHessianFunction(newQuadraticBowlHessian(target))
	x2 := dvector([]float64{5.0, 5.0})
	trnOpts := datamunge.NewTrustRegionNewtonOptions()
	trnOpts.SetMax_iterations(100)
	trnOpts.SetTolerance(1e-10)
	value2 := datamunge.NewTrustRegionNewton(trnOpts).Optimize(trnFn, x2)
	fmt.Printf("TrustRegionNewton: f=%v x=%v (true minimum: f=0 at %v)\n", value2, dvectorSlice(x2), target)

	fmt.Println("\n=================== ResidualFunction: LevenbergMarquardt curve fit ===================")
	trueA, trueK, trueC := 5.0, 0.7, 1.0
	ts := []float64{0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0}
	ys := make([]float64, len(ts))
	for i, t := range ts {
		ys[i] = trueA*math.Exp(-trueK*t) + trueC
	}
	residualFn := datamunge.NewDirectorResidualFunction(&ExpDecayResidual{ts: ts, ys: ys})
	p := dvector([]float64{1.0, 0.1, 0.0})
	lmOpts := datamunge.NewLevenbergMarquardtOptions()
	lmOpts.SetMax_iterations(200)
	lmOpts.SetTolerance(1e-12)
	value3 := datamunge.NewLevenbergMarquardt(lmOpts).Optimize(residualFn, p)
	fmt.Printf("LevenbergMarquardt: sum-sq-residual=%v params(A,k,c)=%v\n", value3, dvectorSlice(p))
	fmt.Printf("                    true params(A,k,c)=[%v, %v, %v]\n", trueA, trueK, trueC)

	fmt.Println("\n=================== BayesianOptimization with RBFGaussianProcessSurrogate ===================")
	boFn := datamunge.NewDirectorArbitraryFunction(Shifted1DBowl{})
	x4 := dvector([]float64{0.0})
	lower := dvector([]float64{-3.0})
	upper := dvector([]float64{5.0})
	surrogate := datamunge.NewRBFGaussianProcessSurrogate(1.0, 1e-6)
	boOpts := datamunge.NewBayesianOptimizationOptions()
	boOpts.SetInitial_samples(10)
	boOpts.SetMax_iterations(60)
	boOpts.SetSeed(42)
	value4 := datamunge.NewBayesianOptimization(boOpts).Optimize(boFn, x4, lower, upper, surrogate.SwigGetBayesianSurrogate())
	fmt.Printf("BayesianOptimization: f=%v x=%v (target minimum near x=1.7)\n", value4, dvectorSlice(x4))
}
