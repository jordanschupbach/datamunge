// CMD: just TARGET=optim_advanced_ex run-go

//go:build ignore

// Advanced datamunge optimizers (ProximalGradient, FISTA, LevenbergMarquardt, Newton, TrustRegionNewton, AugmentedLagrangian, SQP, InteriorPoint, BayesianOptimization, RBFGaussianProcessSurrogate) from Go.
package main

import (
	"datamunge"
	"fmt"
	"math"
	"strings"
)

// Every one of these optimizers takes an interface (ProximalFunction / HessianFunction /
// EqualityConstrainedFunction / InequalityConstrainedFunction / ResidualFunction, or
// ArbitraryFunction for BayesianOptimization) that the C++ library leaves abstract: callers
// implement the objective. From Go, write a type with the virtual methods and wrap it with
// NewXFrom(impl); the optimizer then calls the Go methods. The coordinates are passed as a
// *[]float64 and updated in place.

func check(err error) {
	if err != nil {
		panic(err)
	}
}

// quadratic is f(x, y) = (x - 1)^2 + 10 (y + 2)^2 with exact gradient and Hessian.
type quadratic struct{}

func (quadratic) Evaluate(c []float64) float64 {
	return math.Pow(c[0]-1, 2) + 10*math.Pow(c[1]+2, 2)
}

func (quadratic) Gradient(c []float64) []float64 {
	return []float64{2 * (c[0] - 1), 20 * (c[1] + 2)}
}

func (quadratic) Hessian(c []float64) [][]float64 {
	return [][]float64{{2.0, 0.0}, {0.0, 20.0}}
}

// exponential is the residuals of y = a * exp(b t) against exact data from a = 2, b = 0.5.
type exponential struct {
	ts []float64
}

func (e exponential) Residuals(p []float64) []float64 {
	r := make([]float64, len(e.ts))
	for i, t := range e.ts {
		r[i] = p[0]*math.Exp(p[1]*t) - 2*math.Exp(0.5*t)
	}
	return r
}

func (e exponential) Jacobian(p []float64) [][]float64 {
	var rows [][]float64
	for _, t := range e.ts {
		rows = append(rows, []float64{math.Exp(p[1] * t), p[0] * t * math.Exp(p[1]*t)})
	}
	return rows
}

// incomplete leaves Gradient and Hessian out.
type incomplete struct{}

func (incomplete) Evaluate(c []float64) float64 { return 0.0 }

func main() {
	fmt.Println("=================== RBFGaussianProcessSurrogate (concrete BayesianSurrogate)" +
		" ===================")
	// Fit a 1D surrogate to a simple bowl y = (x-1)^2 sampled at a few points,
	// then check the acquisition function favors an unexplored region.
	xs := []float64{0.0, 0.5, 1.0, 1.5, 2.0, 3.0}
	var points [][]float64
	var ys []float64
	for _, x := range xs {
		points = append(points, []float64{x})
		ys = append(ys, (x-1)*(x-1))
	}
	surrogate := datamunge.Must(datamunge.NewRBFGaussianProcessSurrogateWithNoise(1.0, 1e-6))
	defer surrogate.Close()
	check(surrogate.Fit(points, ys))
	incumbent := 0.0 // best observed so far
	for _, y := range ys {
		incumbent = math.Min(incumbent, y)
	}
	fmt.Printf("incumbent (best observed y): %v\n", incumbent)
	for _, x := range []float64{0.25, 1.0, 2.5} {
		fmt.Printf("acquisition(x=%.2f) = %.6f\n", x, datamunge.Must(surrogate.Acquisition([]float64{x}, incumbent)))
	}

	fmt.Println("\n=================== Options structs + optimizer construction (all 10 new classes)" +
		" ===================")
	// Every new Options struct and optimizer class constructs cleanly, proving
	// the binding for the whole batch is wired correctly.
	pgOptions := datamunge.Must(datamunge.NewProximalGradientOptions())
	check(pgOptions.SetStepSize(0.05))
	pg := datamunge.Must(datamunge.NewProximalGradientWithOptions(pgOptions))
	fista := datamunge.Must(datamunge.NewFISTAWithOptions(datamunge.Must(datamunge.NewFISTAOptions())))
	lmOptions := datamunge.Must(datamunge.NewLevenbergMarquardtOptions())
	check(lmOptions.SetMaxIterations(200))
	lm := datamunge.Must(datamunge.NewLevenbergMarquardtWithOptions(lmOptions))
	newton := datamunge.Must(datamunge.NewNewtonWithOptions(datamunge.Must(datamunge.NewNewtonOptions())))
	trn := datamunge.Must(datamunge.NewTrustRegionNewtonWithOptions(datamunge.Must(datamunge.NewTrustRegionNewtonOptions())))
	al := datamunge.Must(datamunge.NewAugmentedLagrangianWithOptions(datamunge.Must(datamunge.NewAugmentedLagrangianOptions())))
	sqp := datamunge.Must(datamunge.NewSQPWithOptions(datamunge.Must(datamunge.NewSQPOptions())))
	ip := datamunge.Must(datamunge.NewInteriorPointWithOptions(datamunge.Must(datamunge.NewInteriorPointOptions())))
	boOptions := datamunge.Must(datamunge.NewBayesianOptimizationOptions())
	check(boOptions.SetInitialSamples(8))
	bo := datamunge.Must(datamunge.NewBayesianOptimizationWithOptions(boOptions))
	for _, optimizer := range []fmt.Stringer{pg, fista, lm, newton, trn, al, sqp, ip, bo} {
		name := strings.TrimPrefix(fmt.Sprintf("%T", optimizer), "*datamunge.")
		fmt.Printf("%s: %v\n", name, optimizer)
	}

	fmt.Println("\n=================== Go objectives (implementing the C++ interfaces) ===================")
	// Newton's method on the quadratic bowl.
	objective := datamunge.Must(datamunge.NewHessianFunctionFrom(quadratic{}))
	defer objective.Close()
	start := []float64{5.0, 5.0}
	value := datamunge.Must(newton.Optimize(objective, &start))
	fmt.Printf("Newton: f = %.6g at (%.6f, %.6f) (exact minimum 0 at (1, -2))\n", value, start[0], start[1])

	// Levenberg-Marquardt fitting y = a * exp(b t).
	residuals := datamunge.Must(datamunge.NewResidualFunctionFrom(exponential{ts: []float64{0.0, 0.5, 1.0, 1.5, 2.0}}))
	defer residuals.Close()
	params := []float64{1.0, 0.1}
	sse := datamunge.Must(lm.Optimize(residuals, &params))
	fmt.Printf("LevenbergMarquardt: residual = %.3g, a = %.6f, b = %.6f (exact: 2, 0.5)\n", sse, params[0], params[1])

	// An interface method the Go type leaves out is an error when the optimizer calls it.
	missing := datamunge.Must(datamunge.NewHessianFunctionFrom(incomplete{}))
	defer missing.Close()
	_, err := newton.Optimize(missing, &[]float64{0.0, 0.0})
	ok := err == nil
	mentions := err != nil && strings.Contains(err.Error(), "gradient")
	fmt.Printf("missing gradient(): ok=%v, error mentions it: %v\n", ok, mentions)
}
