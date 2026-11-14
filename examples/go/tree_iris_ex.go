// CMD: just TARGET=tree_iris_ex run-go

//go:build ignore

// Decision tree classifier on iris: summary, confusion matrix, misclassified rows and plots.
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

	model := datamunge.Must(datamunge.NewDecisionTreeClassifier(iris, "Species ~ Petal.Length + Petal.Width"))
	defer model.Close()
	check(model.PrintSummary())

	fmt.Println("\nConfusion matrix (rows = actual, cols = predicted):")
	fmt.Println(datamunge.Must(model.ConfusionMatrix()))

	fmt.Println("\nMisclassified rows:")
	predictions := datamunge.Must(model.Predict(iris))
	n := datamunge.Must(iris.Nrows())
	misclassified := 0
	for i, predicted := range predictions {
		actual := datamunge.Must(iris.StringAt("Species", i))
		if predicted == actual {
			continue
		}
		misclassified++
		fmt.Printf("  row %d: Petal.Length=%v Petal.Width=%v  actual=%s  predicted=%s\n",
			i, datamunge.Must(iris.NumericAt("Petal.Length", i)),
			datamunge.Must(iris.NumericAt("Petal.Width", i)), actual, predicted)
	}
	fmt.Printf("%d of %d misclassified (%v%%)\n", misclassified, n, 100.0*float64(misclassified)/float64(n))

	check(datamunge.Must(model.PlotClassification(iris, "Petal.Length", "Petal.Width")).Save("tree_iris_classification.svg"))
	check(datamunge.Must(model.PlotDecisionRegions("Petal.Length", "Petal.Width")).Save("tree_iris_decision_regions.svg"))
	fmt.Println("\nSaved tree_iris_classification.svg and tree_iris_decision_regions.svg")

	// A shallower tree, for comparison, showing a coarser (but still fairly accurate) decision
	// boundary.
	shallow := datamunge.Must(datamunge.NewDecisionTreeClassifierWithMaxDepth(iris, "Species ~ Petal.Length + Petal.Width", 2))
	defer shallow.Close()
	fmt.Printf("\nDepth-2 tree training accuracy: %v%% (%d leaves)\n",
		datamunge.Must(shallow.TrainingAccuracy())*100, datamunge.Must(shallow.LeafCount()))
	check(datamunge.Must(shallow.PlotDecisionRegions("Petal.Length", "Petal.Width")).Save("tree_iris_decision_regions_depth2.svg"))
	fmt.Println("Saved tree_iris_decision_regions_depth2.svg")
}
