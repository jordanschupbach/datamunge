// CMD: just TARGET=xgboost_iris_ex run-go

//go:build ignore

// XGBoost classifier on iris: summary, confusion matrix, misclassified rows, L2-regularization comparison.
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
	iris := datamunge.Must(data.Iris())
	defer iris.Close()
	fmt.Printf("iris: %d rows x %d cols\n\n", datamunge.Must(iris.Nrows()), datamunge.Must(iris.Ncols()))

	model := datamunge.Must(datamunge.NewXGBoostClassifier(iris, "Species ~ Petal.Length + Petal.Width"))
	defer model.Close()
	check(model.PrintSummary())

	fmt.Println("\nConfusion matrix (rows = actual, cols = predicted):\n" + datamunge.Must(model.ConfusionMatrix()).String())

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
			i, datamunge.Must(iris.NumericAt("Petal.Length", i)), datamunge.Must(iris.NumericAt("Petal.Width", i)),
			actual, predicted)
	}
	fmt.Printf("%d of %d misclassified (%v%%)\n", misclassified, n, 100.0*float64(misclassified)/float64(n))

	check(datamunge.Must(model.PlotTrainingDeviance()).Save("xgboost_iris_training_deviance.svg"))
	check(datamunge.Must(model.PlotDecisionRegions("Petal.Length", "Petal.Width")).Save("xgboost_iris_decision_regions.svg"))
	fmt.Println("\nSaved xgboost_iris_training_deviance.svg and xgboost_iris_decision_regions.svg")

	// XGBoost's defining lever is regularization: a heavily L2-regularized model (large lambda)
	// grows the same deep trees but keeps every leaf weight small, producing a much smoother
	// decision boundary than the lightly-regularized default.
	// NewXGBoostClassifierWith<LastParam>(data, formula, nTrees, learningRate, maxDepth, lambda, alpha,
	//     gamma, minChildWeight, minSamplesLeaf, subsample, colsampleBytree, seed)
	heavy := datamunge.Must(datamunge.NewXGBoostClassifierWithLambda(iris, "Species ~ Petal.Length + Petal.Width", 100, 0.3, 6, 50.0))
	defer heavy.Close()
	fmt.Printf("\nlambda=1 (default):   training accuracy=%v%%  deviance=%v\n",
		100*datamunge.Must(model.TrainingAccuracy()), last(datamunge.Must(model.TrainingDeviance())))
	fmt.Printf("lambda=50 (heavy L2): training accuracy=%v%%  deviance=%v\n",
		100*datamunge.Must(heavy.TrainingAccuracy()), last(datamunge.Must(heavy.TrainingDeviance())))
	check(datamunge.Must(heavy.PlotDecisionRegions("Petal.Length", "Petal.Width")).Save("xgboost_iris_decision_regions_heavy_lambda.svg"))
	fmt.Println("Saved xgboost_iris_decision_regions_heavy_lambda.svg")
}
