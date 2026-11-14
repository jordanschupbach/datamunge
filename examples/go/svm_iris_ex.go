// CMD: just TARGET=svm_iris_ex run-go

//go:build ignore

// Support vector machine on iris: RBF vs. linear kernels, confusion matrix and predictions.
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
	formula := "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width"
	iris := datamunge.Must(data.Iris())
	defer iris.Close()
	fmt.Printf("iris: %d rows x %d cols\n\n", datamunge.Must(iris.Nrows()), datamunge.Must(iris.Ncols()))

	fmt.Println("=== RBF kernel (default) ===")
	rbfModel := datamunge.Must(datamunge.NewSVM(iris, formula))
	defer rbfModel.Close()
	fmt.Print(datamunge.Must(rbfModel.Summary()))
	fmt.Println("\nConfusion matrix (rows = actual, cols = predicted):\n" + datamunge.Must(rbfModel.ConfusionMatrix()).String())

	fmt.Println("\n=== Linear kernel, for comparison ===")
	linearModel := datamunge.Must(datamunge.NewSVMWithKernel(iris, formula, "linear"))
	defer linearModel.Close()
	fmt.Printf("Training accuracy: %v%%\n", datamunge.Must(linearModel.TrainingAccuracy())*100)
	fmt.Printf("Support vectors: %d\n", datamunge.Must(linearModel.NumSupportVectors()))

	newdata := datamunge.Must(datamunge.NewDataFrame())
	defer newdata.Close()
	check(newdata.AddNumericColumn("Sepal.Length", []float64{5.1, 6.0, 6.5, 6.2}))
	check(newdata.AddNumericColumn("Sepal.Width", []float64{3.5, 2.7, 3.0, 2.8}))
	check(newdata.AddNumericColumn("Petal.Length", []float64{1.4, 4.5, 5.5, 4.8}))
	check(newdata.AddNumericColumn("Petal.Width", []float64{0.2, 1.5, 2.0, 1.8}))
	fmt.Println("\nRBF predictions for new flowers (votes out of 3 one-vs-one pairs):\n" +
		datamunge.Must(rbfModel.PredictFrame(newdata)).String())
}
