// CMD: just TARGET=lm_ex run-go

//go:build ignore

// Linear model: mpg ~ hp + wt + transmission, its ANOVA table and predictions.
package main

import (
	"datamunge"
	"fmt"
)

func check(err error) {
	if err != nil {
		panic(err)
	}
}

func main() {
	cars := datamunge.Must(datamunge.NewDataFrame())
	defer cars.Close()
	check(cars.AddNumericColumn("hp", []float64{110.0, 110.0, 93.0, 110.0, 175.0, 105.0, 245.0, 62.0, 95.0, 123.0}))
	check(cars.AddNumericColumn("wt", []float64{2.62, 2.875, 2.32, 3.215, 3.44, 3.46, 3.57, 3.19, 3.15, 3.44}))
	check(cars.AddStringColumn("transmission", []string{
		"manual", "manual", "manual", "automatic", "automatic",
		"automatic", "automatic", "automatic", "automatic", "automatic"}))
	check(cars.AddNumericColumn("mpg", []float64{21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2}))

	fmt.Println("Fitting: mpg ~ hp + wt + transmission")
	fmt.Println()
	model := datamunge.Must(datamunge.NewLM(cars, "mpg ~ hp + wt + transmission"))
	defer model.Close()
	// PrintSummary writes to C++ std::cout; the binding flushes it, so it stays in order with fmt.
	check(model.PrintSummary())

	fmt.Println("\nSequential ANOVA:")
	fmt.Println(datamunge.Must(model.Anova()))

	newcars := datamunge.Must(datamunge.NewDataFrame())
	defer newcars.Close()
	check(newcars.AddNumericColumn("hp", []float64{150.0, 90.0}))
	check(newcars.AddNumericColumn("wt", []float64{3.0, 2.5}))
	check(newcars.AddStringColumn("transmission", []string{"manual", "automatic"}))
	fmt.Println("\nPredictions with 95% confidence intervals:")
	fmt.Println(datamunge.Must(model.PredictFrameWithIntervalKind(newcars, "confidence")))

	check(model.SaveDiagnosticPlots("lm_ex_diagnostics"))
	fmt.Println("\nSaved diagnostic plots as lm_ex_diagnostics_*.svg")
}
