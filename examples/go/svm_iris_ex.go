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

	fmt.Println("=== RBF kernel (default) ===")
	rbfModel := datamunge.NewSVM(iris, "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width")
	rbfModel.Print_summary()
	fmt.Println("\nConfusion matrix (rows = actual, cols = predicted):")
	fmt.Println(rbfModel.Confusion_matrix().To_string())

	fmt.Println("\n=== Linear kernel, for comparison ===")
	linearModel := datamunge.NewSVM(iris, "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width", "linear")
	fmt.Printf("Training accuracy: %g%%\n", linearModel.Training_accuracy()*100.0)
	fmt.Printf("Support vectors: %d\n", linearModel.Num_support_vectors())

	newdata := datamunge.NewDataFrame()
	newdata.Add_numeric_column("Sepal.Length", dvector([]float64{5.1, 6.0, 6.5, 6.2}))
	newdata.Add_numeric_column("Sepal.Width", dvector([]float64{3.5, 2.7, 3.0, 2.8}))
	newdata.Add_numeric_column("Petal.Length", dvector([]float64{1.4, 4.5, 5.5, 4.8}))
	newdata.Add_numeric_column("Petal.Width", dvector([]float64{0.2, 1.5, 2.0, 1.8}))

	fmt.Println("\nRBF predictions for new flowers (votes out of 3 one-vs-one pairs):")
	fmt.Println(rbfModel.Predict_frame(newdata).To_string())
}
