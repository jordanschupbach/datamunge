// CMD: just TARGET=lda_iris_ex run-go

//go:build ignore

// Linear discriminant analysis on iris: summary, confusion matrix, predictions and discriminant plot.
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
	iris := datamunge.Must(data.Iris())
	defer iris.Close()
	fmt.Printf("iris: %d rows x %d cols\n\n", datamunge.Must(iris.Nrows()), datamunge.Must(iris.Ncols()))

	model := datamunge.Must(datamunge.NewLDA(iris, "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width"))
	defer model.Close()
	check(model.PrintSummary())

	fmt.Println("\nConfusion matrix (rows = actual, cols = predicted):\n" + datamunge.Must(model.ConfusionMatrix()).String())

	newdata := datamunge.Must(datamunge.NewDataFrame())
	defer newdata.Close()
	check(newdata.AddNumericColumn("Sepal.Length", []float64{5.1, 6.0, 6.5, 6.2}))
	check(newdata.AddNumericColumn("Sepal.Width", []float64{3.5, 2.7, 3.0, 2.8}))
	check(newdata.AddNumericColumn("Petal.Length", []float64{1.4, 4.5, 5.5, 4.8}))
	check(newdata.AddNumericColumn("Petal.Width", []float64{0.2, 1.5, 2.0, 1.8}))
	fmt.Println("\nPredictions for new flowers:\n" + datamunge.Must(model.PredictFrame(newdata)).String())

	check(model.SaveDiscriminantPlot("lda_iris_discriminants.svg"))
	fmt.Println("\nSaved discriminant plot as lda_iris_discriminants.svg")
}
