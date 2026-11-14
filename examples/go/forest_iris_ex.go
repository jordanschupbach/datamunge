// CMD: just TARGET=forest_iris_ex run-go

//go:build ignore

// Random forest classifier on iris: summary, confusion matrix, misclassifications, OOB accuracy.
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

	model := datamunge.Must(datamunge.NewRandomForestClassifier(iris, "Species ~ Petal.Length + Petal.Width"))
	defer model.Close()
	// PrintSummary writes to C++ std::cout; the binding flushes it, so it stays in order with fmt.
	check(model.PrintSummary())

	fmt.Println("\nConfusion matrix (rows = actual, cols = predicted):")
	fmt.Println(datamunge.Must(model.ConfusionMatrix()))

	fmt.Println("\nMisclassified rows:")
	predictions := datamunge.Must(model.Predict(iris))
	n := datamunge.Must(iris.Nrows())
	misclassified := 0
	for i := 0; i < n; i++ {
		actual := datamunge.Must(iris.StringAt("Species", i))
		predicted := predictions[i]
		if predicted == actual {
			continue
		}
		misclassified++
		fmt.Printf("  row %d: Petal.Length=%v Petal.Width=%v  actual=%s  predicted=%s\n",
			i, datamunge.Must(iris.NumericAt("Petal.Length", i)),
			datamunge.Must(iris.NumericAt("Petal.Width", i)), actual, predicted)
	}
	fmt.Printf("%d of %d misclassified (%v%%)\n", misclassified, n, 100.0*float64(misclassified)/float64(n))

	check(datamunge.Must(model.PlotClassification(iris, "Petal.Length", "Petal.Width")).Save("forest_iris_classification.svg"))
	check(datamunge.Must(model.PlotDecisionRegions("Petal.Length", "Petal.Width")).Save("forest_iris_decision_regions.svg"))
	fmt.Println("\nSaved forest_iris_classification.svg and forest_iris_decision_regions.svg")

	// A small forest, for comparison, showing how out-of-bag accuracy stabilizes as more trees
	// are added -- the defining random forest effect a single decision tree cannot demonstrate.
	// NewRandomForestClassifierWithSeed(data, formula, nTrees, maxDepth, minSamplesSplit,
	// minSamplesLeaf, maxFeatures, criterion, bootstrap, sampleFraction, seed) takes every option.
	smallForest := datamunge.Must(datamunge.NewRandomForestClassifierWithNTrees(iris, "Species ~ Petal.Length + Petal.Width", 5))
	defer smallForest.Close()
	fmt.Printf("\n5-tree forest:   training accuracy=%v%%  OOB accuracy=%v%%\n",
		datamunge.Must(smallForest.TrainingAccuracy())*100, datamunge.Must(smallForest.OobAccuracy())*100)
	fmt.Printf("100-tree forest: training accuracy=%v%%  OOB accuracy=%v%%\n",
		datamunge.Must(model.TrainingAccuracy())*100, datamunge.Must(model.OobAccuracy())*100)
	check(datamunge.Must(smallForest.PlotDecisionRegions("Petal.Length", "Petal.Width")).Save("forest_iris_decision_regions_5trees.svg"))
	fmt.Println("Saved forest_iris_decision_regions_5trees.svg")
}
