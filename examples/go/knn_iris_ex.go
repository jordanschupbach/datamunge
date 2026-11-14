// CMD: just TARGET=knn_iris_ex run-go

//go:build ignore

// k-nearest-neighbours classifier on iris: leave-one-out accuracy and decision regions for several k.
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
	formula := "Species ~ Petal.Length + Petal.Width"

	iris := datamunge.Must(data.Iris())
	defer iris.Close()
	fmt.Printf("iris: %d rows x %d cols\n\n", datamunge.Must(iris.Nrows()), datamunge.Must(iris.Ncols()))

	model := datamunge.Must(datamunge.NewKNNClassifier(iris, formula))
	defer model.Close()
	// Summary returns the text PrintSummary writes to native stdout, so printing it with fmt
	// keeps it in order with the surrounding output.
	fmt.Print(datamunge.Must(model.Summary()))

	fmt.Println("\nConfusion matrix (leave-one-out; rows = actual, cols = predicted):")
	fmt.Println(datamunge.Must(model.ConfusionMatrix()))

	fmt.Println("\nLeave-one-out misclassified rows:")
	fitted := datamunge.Must(model.Predict(iris))
	n := datamunge.Must(iris.Nrows())
	misclassified := 0
	for i := 0; i < n; i++ {
		actual := datamunge.Must(iris.StringAt("Species", i))
		predicted := fitted[i]
		if predicted == actual {
			continue
		}
		misclassified++
		fmt.Printf("  row %d: Petal.Length=%v Petal.Width=%v  actual=%s  predicted=%s\n",
			i, datamunge.Must(iris.NumericAt("Petal.Length", i)),
			datamunge.Must(iris.NumericAt("Petal.Width", i)), actual, predicted)
	}
	fmt.Printf("%d of %d misclassified (%v%%)\n", misclassified, n, 100.0*float64(misclassified)/float64(n))

	check(datamunge.Must(model.PlotDecisionRegions("Petal.Length", "Petal.Width")).Save("knn_iris_decision_regions_k5.svg"))
	fmt.Println("\nSaved knn_iris_decision_regions_k5.svg")

	// k=1 memorizes every training point exactly (jagged, overfit boundary with an island around
	// every point, including noise); k=25 averages over a much larger neighborhood (very smooth,
	// underfit boundary). NewKNNClassifierWithStandardize(data, formula, k, metric, weighted,
	// standardize) takes every option.
	for _, k := range []int{1, 25} {
		knn := datamunge.Must(datamunge.NewKNNClassifierWithK(iris, formula, k))
		file := fmt.Sprintf("knn_iris_decision_regions_k%d.svg", k)
		fmt.Printf("\nk=%-2d leave-one-out accuracy: %v%%\n", k, datamunge.Must(knn.TrainingAccuracy())*100)
		check(datamunge.Must(knn.PlotDecisionRegions("Petal.Length", "Petal.Width")).Save(file))
		fmt.Println("Saved " + file)
		knn.Close()
	}
}
