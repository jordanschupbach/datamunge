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

func main() {
	FORMULA := "Petal.Length ~ Petal.Width"

	iris := datamunge.DataFrameIris()
	fmt.Printf("iris: %d rows x %d cols\n", iris.Nrows(), iris.Ncols())
	fmt.Printf("formula: %s\n\n", FORMULA)

	model := datamunge.NewGaussianProcessRegression(iris, FORMULA)
	model.Print_summary()

	model.Plot_fit(iris).Save("gpr_iris_fit.svg")
	model.Plot_length_scale_profile().Save("gpr_iris_length_scale_profile.svg")
	fmt.Println("\nSaved gpr_iris_fit.svg and gpr_iris_length_scale_profile.svg")

	query := datamunge.NewDataFrame()
	query.Add_numeric_column("Petal.Width", dvector([]float64{0.2, 1.3, 2.5, 10.0}))
	detail := model.Predict_frame(query, "confidence")
	fmt.Println("\nPredictions with 95% confidence intervals:")
	fmt.Println(detail.To_string())
	fmt.Println("(Petal.Width=10.0 is far outside the training range [0.1, 2.5] -- note how much wider its\n interval is than the in-range predictions.)")
}
