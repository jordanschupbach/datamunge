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
	iris := datamunge.DataFrameIris()
	fmt.Printf("iris: %d rows x %d cols\n\n", iris.Nrows(), iris.Ncols())

	model := datamunge.NewLDA(iris, "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width")
	model.Print_summary()

	fmt.Println("\nConfusion matrix (rows = actual, cols = predicted):")
	fmt.Println(model.Confusion_matrix().To_string())

	newdata := datamunge.NewDataFrame()
	newdata.Add_numeric_column("Sepal.Length", dvector([]float64{5.1, 6.0, 6.5, 6.2}))
	newdata.Add_numeric_column("Sepal.Width", dvector([]float64{3.5, 2.7, 3.0, 2.8}))
	newdata.Add_numeric_column("Petal.Length", dvector([]float64{1.4, 4.5, 5.5, 4.8}))
	newdata.Add_numeric_column("Petal.Width", dvector([]float64{0.2, 1.5, 2.0, 1.8}))

	fmt.Println("\nPredictions for new flowers:")
	fmt.Println(model.Predict_frame(newdata).To_string())

	model.Save_discriminant_plot("lda_iris_discriminants.svg")
	fmt.Println("\nSaved discriminant plot as lda_iris_discriminants.svg")
}
