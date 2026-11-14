// CMD: just TARGET=gaussian_process_regression_iris_ex run-go

//go:build ignore

// Gaussian process regression of Petal.Length on Petal.Width (iris), with confidence intervals.
package main

import (
	"datamunge"
	"datamunge/data"
	"fmt"
)

func check(err error) {
	if err != nil {
		panic(err)
	}
}

func main() {
	formula := "Petal.Length ~ Petal.Width"

	iris := datamunge.Must(data.Iris())
	defer iris.Close()
	fmt.Printf("iris: %d rows x %d cols\n", datamunge.Must(iris.Nrows()), datamunge.Must(iris.Ncols()))
	fmt.Printf("formula: %s\n\n", formula)

	// Both the length scale and the noise ratio are auto-selected by maximizing the exact log
	// marginal likelihood.
	model := datamunge.Must(datamunge.NewGaussianProcessRegression(iris, formula))
	defer model.Close()
	fmt.Print(datamunge.Must(model.Summary()))

	check(datamunge.Must(model.PlotFit(iris)).Save("gpr_iris_fit.svg"))
	check(datamunge.Must(model.PlotLengthScaleProfile()).Save("gpr_iris_length_scale_profile.svg"))
	fmt.Println("\nSaved gpr_iris_fit.svg and gpr_iris_length_scale_profile.svg")

	// Unlike every other regressor in this suite, a GP gives a genuine posterior confidence
	// interval at every point -- including far outside the training data, where it should widen
	// substantially as the model's uncertainty grows.
	query := datamunge.Must(datamunge.NewDataFrame())
	defer query.Close()
	check(query.AddNumericColumn("Petal.Width", []float64{0.2, 1.3, 2.5, 10.0}))
	detail := datamunge.Must(model.PredictFrameWithIntervalKind(query, "confidence"))
	fmt.Println("\nPredictions with 95% confidence intervals:")
	fmt.Println(detail)
	fmt.Println("(Petal.Width=10.0 is far outside the training range [0.1, 2.5] -- note how much wider" +
		" its\n interval is than the in-range predictions.)")
}
