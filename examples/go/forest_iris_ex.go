package main

import (
	"datamunge"
	"fmt"
)

func main() {
	iris := datamunge.DataFrameIris()
	fmt.Printf("iris: %d rows x %d cols\n\n", iris.Nrows(), iris.Ncols())

	model := datamunge.NewRandomForestClassifier(iris, "Species ~ Petal.Length + Petal.Width")
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

	model.Plot_classification(iris, "Petal.Length", "Petal.Width").Save("forest_iris_classification.svg")
	model.Plot_decision_regions("Petal.Length", "Petal.Width").Save("forest_iris_decision_regions.svg")
	fmt.Println("\nSaved forest_iris_classification.svg and forest_iris_decision_regions.svg")

	smallForest := datamunge.NewRandomForestClassifier(iris, "Species ~ Petal.Length + Petal.Width", int64(5))
	fmt.Printf("\n5-tree forest:   training accuracy=%g%%  OOB accuracy=%g%%\n", smallForest.Training_accuracy()*100.0, smallForest.Oob_accuracy()*100.0)
	fmt.Printf("100-tree forest: training accuracy=%g%%  OOB accuracy=%g%%\n", model.Training_accuracy()*100.0, model.Oob_accuracy()*100.0)
	smallForest.Plot_decision_regions("Petal.Length", "Petal.Width").Save("forest_iris_decision_regions_5trees.svg")
	fmt.Println("Saved forest_iris_decision_regions_5trees.svg")
}
