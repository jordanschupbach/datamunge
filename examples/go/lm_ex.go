package main

import (
	"datamunge"
	"fmt"
)

func dvector(values []float64) datamunge.DVector {
	out := datamunge.NewDVector(int64(len(values)))
	for i, v := range values {
		out.Set(i, v)
	}
	return out
}

func svector(values []string) datamunge.SVector {
	out := datamunge.NewSVector(int64(len(values)))
	for i, v := range values {
		out.Set(i, v)
	}
	return out
}

func main() {
	hp := []float64{110.0, 110.0, 93.0, 110.0, 175.0, 105.0, 245.0, 62.0, 95.0, 123.0}
	wt := []float64{2.62, 2.875, 2.32, 3.215, 3.44, 3.46, 3.57, 3.19, 3.15, 3.44}
	transmission := []string{"manual", "manual", "manual", "automatic", "automatic", "automatic", "automatic", "automatic", "automatic", "automatic"}
	mpg := []float64{21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2}

	cars := datamunge.NewDataFrame()
	cars.Add_numeric_column("hp", dvector(hp))
	cars.Add_numeric_column("wt", dvector(wt))
	cars.Add_string_column("transmission", svector(transmission))
	cars.Add_numeric_column("mpg", dvector(mpg))

	fmt.Println("Fitting: mpg ~ hp + wt + transmission")
	fmt.Println()
	model := datamunge.NewLM(cars, "mpg ~ hp + wt + transmission")
	model.Print_summary()

	fmt.Println("\nSequential ANOVA:")
	fmt.Println(model.Anova().To_string())

	newcars := datamunge.NewDataFrame()
	newcars.Add_numeric_column("hp", dvector([]float64{150.0, 90.0}))
	newcars.Add_numeric_column("wt", dvector([]float64{3.0, 2.5}))
	newcars.Add_string_column("transmission", svector([]string{"manual", "automatic"}))

	frame := model.Predict_frame(newcars, "confidence")
	fmt.Println("\nPredictions with 95% confidence intervals:")
	fmt.Println(frame.To_string())

	model.Save_diagnostic_plots("lm_ex_diagnostics")
	fmt.Println("\nSaved diagnostic plots as lm_ex_diagnostics_*.svg")
}
