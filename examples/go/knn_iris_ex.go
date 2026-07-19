package main

import (
	"datamunge"
	"fmt"
)

func main() {
	iris := datamunge.DataFrameIris()
	fmt.Printf("iris: %d rows x %d cols\n\n", iris.Nrows(), iris.Ncols())

	model := datamunge.NewKNNClassifier(iris, "Species ~ Petal.Length + Petal.Width")
	model.Print_summary()

	fmt.Println("\nConfusion matrix (leave-one-out; rows = actual, cols = predicted):")
	fmt.Println(model.Confusion_matrix().To_string())

	fmt.Println("\nLeave-one-out misclassified rows:")
	fitted := model.Predict(iris)
	misclassified := 0
	n := iris.Nrows()
	for i := int64(0); i < n; i++ {
		actual := iris.String_at("Species", i)
		pred := fitted.Get(int(i))
		if pred != actual {
			misclassified++
			fmt.Printf("  row %d: Petal.Length=%g Petal.Width=%g  actual=%s  predicted=%s\n", i,
				iris.Numeric_at("Petal.Length", i), iris.Numeric_at("Petal.Width", i), actual, pred)
		}
	}
	fmt.Printf("%d of %d misclassified (%g%%)\n", misclassified, n, 100.0*float64(misclassified)/float64(n))

	model.Plot_decision_regions("Petal.Length", "Petal.Width").Save("knn_iris_decision_regions_k5.svg")
	fmt.Println("\nSaved knn_iris_decision_regions_k5.svg")

	k1 := datamunge.NewKNNClassifier(iris, "Species ~ Petal.Length + Petal.Width", int64(1))
	fmt.Printf("\nk=1  leave-one-out accuracy: %g%%\n", k1.Training_accuracy()*100.0)
	k1.Plot_decision_regions("Petal.Length", "Petal.Width").Save("knn_iris_decision_regions_k1.svg")
	fmt.Println("Saved knn_iris_decision_regions_k1.svg")

	k25 := datamunge.NewKNNClassifier(iris, "Species ~ Petal.Length + Petal.Width", int64(25))
	fmt.Printf("\nk=25 leave-one-out accuracy: %g%%\n", k25.Training_accuracy()*100.0)
	k25.Plot_decision_regions("Petal.Length", "Petal.Width").Save("knn_iris_decision_regions_k25.svg")
	fmt.Println("Saved knn_iris_decision_regions_k25.svg")
}
