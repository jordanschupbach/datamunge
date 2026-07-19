package main

import (
	"datamunge"
	"fmt"
)

func main() {
	iris := datamunge.DataFrameIris()
	fmt.Printf("iris: %d rows x %d cols\n\n", iris.Nrows(), iris.Ncols())

	model := datamunge.NewDecisionTreeClassifier(iris, "Species ~ Petal.Length + Petal.Width")
	model.Print_summary()

	fmt.Println("\nConfusion matrix (rows = actual, cols = predicted):")
	fmt.Println(model.Confusion_matrix().To_string())

	fmt.Println("\nMisclassified rows:")
	predictions := model.Predict(iris)
	misclassified := 0
	n := iris.Nrows()
	for i := int64(0); i < n; i++ {
		actual := iris.String_at("Species", i)
		pred := predictions.Get(int(i))
		if pred != actual {
			misclassified++
			fmt.Printf("  row %d: Petal.Length=%g Petal.Width=%g  actual=%s  predicted=%s\n", i,
				iris.Numeric_at("Petal.Length", i), iris.Numeric_at("Petal.Width", i), actual, pred)
		}
	}
	fmt.Printf("%d of %d misclassified (%g%%)\n", misclassified, n, 100.0*float64(misclassified)/float64(n))

	model.Plot_classification(iris, "Petal.Length", "Petal.Width").Save("tree_iris_classification.svg")
	model.Plot_decision_regions("Petal.Length", "Petal.Width").Save("tree_iris_decision_regions.svg")
	fmt.Println("\nSaved tree_iris_classification.svg and tree_iris_decision_regions.svg")

	shallow := datamunge.NewDecisionTreeClassifier(iris, "Species ~ Petal.Length + Petal.Width", int64(2))
	fmt.Printf("\nDepth-2 tree training accuracy: %g%% (%d leaves)\n", shallow.Training_accuracy()*100.0, shallow.Leaf_count())
	shallow.Plot_decision_regions("Petal.Length", "Petal.Width").Save("tree_iris_decision_regions_depth2.svg")
	fmt.Println("Saved tree_iris_decision_regions_depth2.svg")
}
