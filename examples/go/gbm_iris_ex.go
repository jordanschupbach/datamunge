// CMD: just TARGET=gbm_iris_ex run-go

//go:build ignore

// Gradient-boosted trees (GBM) on iris: summary, confusion matrix, misclassified rows, 5 vs 100 rounds.
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

func last(values []float64) float64 {
	return values[len(values)-1]
}

func main() {
	formula := "Species ~ Petal.Length + Petal.Width"

	iris := datamunge.Must(data.Iris())
	defer iris.Close()
	fmt.Printf("iris: %d rows x %d cols\n\n", datamunge.Must(iris.Nrows()), datamunge.Must(iris.Ncols()))

	model := datamunge.Must(datamunge.NewGBMClassifier(iris, formula))
	defer model.Close()
	// Summary returns the text PrintSummary writes to native stdout.
	fmt.Print(datamunge.Must(model.Summary()))

	fmt.Println("\nConfusion matrix (rows = actual, cols = predicted):\n" +
		datamunge.Must(model.ConfusionMatrix()).String())

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

	check(datamunge.Must(model.PlotTrainingDeviance()).Save("gbm_iris_training_deviance.svg"))
	check(datamunge.Must(model.PlotDecisionRegions("Petal.Length", "Petal.Width")).Save("gbm_iris_decision_regions.svg"))
	fmt.Println("\nSaved gbm_iris_training_deviance.svg and gbm_iris_decision_regions.svg")

	// A handful of boosting rounds vs a well-boosted ensemble: each additional round chips away
	// at the training loss, gradually sharpening the decision boundary.
	// NewGBMClassifierWithSeed(data, formula, nTrees, learningRate, maxDepth, minSamplesSplit,
	//     minSamplesLeaf, subsample, seed)
	few := datamunge.Must(datamunge.NewGBMClassifierWithNTrees(iris, formula, 5))
	defer few.Close()
	fmt.Printf("\n5-round ensemble:   training accuracy=%v%%  deviance=%v\n",
		100*datamunge.Must(few.TrainingAccuracy()), last(datamunge.Must(few.TrainingDeviance())))
	fmt.Printf("100-round ensemble: training accuracy=%v%%  deviance=%v\n",
		100*datamunge.Must(model.TrainingAccuracy()), last(datamunge.Must(model.TrainingDeviance())))
	check(datamunge.Must(few.PlotDecisionRegions("Petal.Length", "Petal.Width")).Save("gbm_iris_decision_regions_5rounds.svg"))
	fmt.Println("Saved gbm_iris_decision_regions_5rounds.svg")
}
