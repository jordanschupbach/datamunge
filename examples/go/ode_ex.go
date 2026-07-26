package main

import (
	"datamunge"
	"fmt"
	"math"
	"strings"
)

func dvector(values []float64) datamunge.DVector {
	out := datamunge.NewDVector(int64(len(values)))
	for i, v := range values {
		out.Set(i, v)
	}
	return out
}

func lastState(sol datamunge.ODESolution) datamunge.DVector {
	return sol.State_at(sol.Size() - 1)
}

func stateString(st datamunge.DVector) string {
	parts := make([]string, st.Size())
	for i := 0; i < int(st.Size()); i++ {
		parts[i] = fmt.Sprintf("%g", st.Get(i))
	}
	return strings.Join(parts, ", ")
}

// NOTE: the Python/Ruby ode examples also demonstrate a live, user-supplied RHS by subclassing
// datamunge.RHS (a SWIG director). This Go example follows the director-free path used by the
// Lua template and uses the built-in named systems via solve_builtin().

func main() {
	fmt.Println("=================== Every named built-in system (no director needed) ===================")
	solver := datamunge.NewODESolver()
	type system struct {
		name   string
		params []float64
		y0     []float64
	}
	systems := []system{
		{"exponential_decay", []float64{1.0}, []float64{1.0}},
		{"logistic_growth", []float64{1.0, 1.0}, []float64{0.5}},
		{"harmonic_oscillator", []float64{1.0}, []float64{1.0, 0.0}},
		{"van_der_pol", []float64{1.0}, []float64{2.0, 0.0}},
		{"lorenz", []float64{10.0, 28.0, 8.0 / 3.0}, []float64{1.0, 1.0, 1.0}},
	}
	for _, s := range systems {
		sol := solver.Solve_builtin(s.name, dvector(s.params), dvector(s.y0), 0.0, 1.0)
		fmt.Printf("%-22s steps=%-6d final state=[%s]\n", s.name, sol.GetSteps_taken(), stateString(lastState(sol)))
	}

	fmt.Println("\n=================== Harmonic oscillator (energy conservation) ===================")
	options := datamunge.NewODEOptions()
	options.SetMethod(datamunge.StepMethod_RK4)
	options.SetStep_size(0.01)
	solver = datamunge.NewODESolver(options)
	sol := solver.Solve_builtin("harmonic_oscillator", dvector([]float64{1.0}), dvector([]float64{1.0, 0.0}), 0.0, 20.0)
	final := lastState(sol)
	x, v := final.Get(0), final.Get(1)
	fmt.Printf("x(20) = %.6f (cos(20) = %.6f)\n", x, math.Cos(20))
	fmt.Printf("energy x^2+v^2 = %.6f (should stay near 1.0)\n", x*x+v*v)

	tSeries := make([]float64, sol.Size())
	xSeries := make([]float64, sol.Size())
	vSeries := make([]float64, sol.Size())
	for i := 0; i < int(sol.Size()); i++ {
		tSeries[i] = sol.Time_at(int64(i))
		st := sol.State_at(int64(i))
		xSeries[i] = st.Get(0)
		vSeries[i] = st.Get(1)
	}
	plot := datamunge.RPlotPlot(dvector(tSeries), dvector(xSeries), "l", "x(t)")
	plot.Lines(dvector(tSeries), dvector(vSeries), "v(t)")
	plot.Title("Harmonic Oscillator").X_label("t").Y_label("state")
	plot.Save_svg("ode_harmonic_oscillator_go.svg")
	fmt.Println("wrote ode_harmonic_oscillator_go.svg")

	fmt.Println("\n=================== Lorenz attractor (phase plane) ===================")
	options = datamunge.NewODEOptions()
	options.SetMethod(datamunge.StepMethod_RK4)
	options.SetStep_size(0.005)
	solver = datamunge.NewODESolver(options)
	sol = solver.Solve_builtin("lorenz", dvector([]float64{10.0, 28.0, 8.0 / 3.0}), dvector([]float64{1.0, 1.0, 1.0}), 0.0, 25.0)
	fmt.Printf("steps_taken = %d\n", sol.GetSteps_taken())

	xs := make([]float64, sol.Size())
	zs := make([]float64, sol.Size())
	for i := 0; i < int(sol.Size()); i++ {
		st := sol.State_at(int64(i))
		xs[i] = st.Get(0)
		zs[i] = st.Get(2)
	}
	plot = datamunge.RPlotPlot(dvector(xs), dvector(zs), "l", "trajectory")
	plot.Title("Lorenz Attractor (x-z phase plane)").X_label("x").Y_label("z")
	plot.Save_svg("ode_lorenz_phase_plane_go.svg")
	fmt.Println("wrote ode_lorenz_phase_plane_go.svg")
}
