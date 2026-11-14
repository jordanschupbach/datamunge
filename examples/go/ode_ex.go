// CMD: just TARGET=ode_ex run-go

//go:build ignore

// ODEs: built-in named systems, harmonic-oscillator energy check and a Lorenz phase plot.
package main

import (
	"datamunge"
	"fmt"
	"math"
	"strings"
)

func check(err error) {
	if err != nil {
		panic(err)
	}
}

func lastState(sol *datamunge.ODESolution) []float64 {
	return datamunge.Must(sol.StateAt(datamunge.Must(sol.Size()) - 1))
}

// The times of every step.
func times(sol *datamunge.ODESolution) []float64 {
	out := make([]float64, datamunge.Must(sol.Size()))
	for i := range out {
		out[i] = datamunge.Must(sol.TimeAt(i))
	}
	return out
}

// One state component (0-based) at every step.
func component(sol *datamunge.ODESolution, c int) []float64 {
	out := make([]float64, datamunge.Must(sol.Size()))
	for i := range out {
		out[i] = datamunge.Must(sol.StateAt(i))[c]
	}
	return out
}

func rk4Solver(step float64) *datamunge.ODESolver {
	options := datamunge.Must(datamunge.NewODEOptions())
	defer options.Close()
	check(options.SetMethod(datamunge.StepMethodRK4))
	check(options.SetStepSize(step))
	return datamunge.Must(datamunge.NewODESolverWithOptions(options))
}

type namedSystem struct {
	name   string
	params []float64
	y0     []float64
}

func main() {
	// NOTE: like the Lua version this example uses the built-in named systems via SolveBuiltin.

	fmt.Println("=================== Every named built-in system (no director needed) ===================")
	solver := datamunge.Must(datamunge.NewODESolver())
	defer solver.Close()
	systems := []namedSystem{
		{"exponential_decay", []float64{1.0}, []float64{1.0}},
		{"logistic_growth", []float64{1.0, 1.0}, []float64{0.5}},
		{"harmonic_oscillator", []float64{1.0}, []float64{1.0, 0.0}},
		{"van_der_pol", []float64{1.0}, []float64{2.0, 0.0}},
		{"lorenz", []float64{10.0, 28.0, 8.0 / 3}, []float64{1.0, 1.0, 1.0}},
	}
	for _, system := range systems {
		sol := datamunge.Must(solver.SolveBuiltin(system.name, system.params, system.y0, 0.0, 1.0))
		parts := []string{}
		for _, v := range lastState(sol) {
			parts = append(parts, fmt.Sprintf("%.6g", v))
		}
		fmt.Printf("%-22s steps=%-6d final state=[%s]\n", system.name, sol.StepsTaken(), strings.Join(parts, ", "))
		sol.Close()
	}

	fmt.Println("\n=================== Harmonic oscillator (energy conservation) ===================")
	rk4 := rk4Solver(0.01)
	defer rk4.Close()
	sol := datamunge.Must(rk4.SolveBuiltin("harmonic_oscillator", []float64{1.0}, []float64{1.0, 0.0}, 0.0, 20.0))
	defer sol.Close()
	last := lastState(sol)
	x, v := last[0], last[1]
	fmt.Printf("x(20) = %.6f (cos(20) = %.6f)\n", x, math.Cos(20.0))
	fmt.Printf("energy x^2+v^2 = %.6f (should stay near 1.0)\n", x*x+v*v)
	ts := times(sol)
	p := datamunge.Must(datamunge.RPlotPlotWithLabel(ts, component(sol, 0), "l", "x(t)"))
	defer p.Close()
	datamunge.Must(p.LinesWithLabel(ts, component(sol, 1), "v(t)"))
	datamunge.Must(p.Title("Harmonic Oscillator"))
	datamunge.Must(p.XLabel("t"))
	datamunge.Must(p.YLabel("state"))
	check(p.SaveSvg("ode_harmonic_oscillator_go.svg"))
	fmt.Println("wrote ode_harmonic_oscillator_go.svg")

	fmt.Println("\n=================== Lorenz attractor (phase plane) ===================")
	rk4Fine := rk4Solver(0.005)
	defer rk4Fine.Close()
	lorenz := datamunge.Must(rk4Fine.SolveBuiltin("lorenz", []float64{10.0, 28.0, 8.0 / 3}, []float64{1.0, 1.0, 1.0}, 0.0, 25.0))
	defer lorenz.Close()
	fmt.Printf("steps_taken = %d\n", lorenz.StepsTaken())
	phase := datamunge.Must(datamunge.RPlotPlotWithLabel(component(lorenz, 0), component(lorenz, 2), "l", "trajectory"))
	defer phase.Close()
	datamunge.Must(phase.Title("Lorenz Attractor (x-z phase plane)"))
	datamunge.Must(phase.XLabel("x"))
	datamunge.Must(phase.YLabel("z"))
	check(phase.SaveSvg("ode_lorenz_phase_plane_go.svg"))
	fmt.Println("wrote ode_lorenz_phase_plane_go.svg")
}
