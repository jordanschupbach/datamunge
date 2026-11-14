// CMD: just TARGET=naive_bayes_penguins_ex run-go

//go:build ignore

// Naive Bayes on penguins: mixed numeric/categorical predictors, confusion matrix, misclassified rows, plots.
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
	// Note: NaiveBayesClassifier drops incomplete rows for its own formula columns internally
	// when fitting, matching every other formula-based model in this library.
	penguins := datamunge.Must(data.Penguins())
	defer penguins.Close()
	nrows := datamunge.Must(penguins.Nrows())
	fmt.Printf("penguins: %d rows x %d cols\n\n", nrows, datamunge.Must(penguins.Ncols()))

	// A mix of numeric (Gaussian likelihood) and categorical (frequency-table likelihood)
	// predictors in one formula -- each modeled independently given the class, per the naive
	// Bayes assumption.
	model := datamunge.Must(datamunge.NewNaiveBayesClassifier(
		penguins, "species ~ bill_length_mm + bill_depth_mm + island + sex"))
	defer model.Close()
	check(model.PrintSummary())

	fmt.Printf("\nConfusion matrix (rows = actual, cols = predicted):\n%v\n", datamunge.Must(model.ConfusionMatrix()))

	fmt.Println("\nMisclassified rows:")
	predictions := datamunge.Must(model.Predict(penguins))
	misclassified := 0
	columns := []string{"species", "bill_length_mm", "bill_depth_mm", "island", "sex"}
rows:
	for i := 0; i < nrows; i++ {
		for _, column := range columns {
			if datamunge.Must(penguins.IsNull(column, i)) {
				continue rows
			}
		}
		actual := datamunge.Must(penguins.StringAt("species", i))
		predicted := predictions[i]
		if predicted == actual {
			continue
		}
		misclassified++
		fmt.Printf("  row %d: bill_length=%v bill_depth=%v island=%s sex=%s  actual=%s  predicted=%s\n",
			i,
			datamunge.Must(penguins.NumericAt("bill_length_mm", i)),
			datamunge.Must(penguins.NumericAt("bill_depth_mm", i)),
			datamunge.Must(penguins.StringAt("island", i)),
			datamunge.Must(penguins.StringAt("sex", i)),
			actual, predicted)
	}
	fmt.Printf("%d misclassified (of %d rows, some incomplete)\n", misclassified, nrows)

	// plot_decision_regions requires exactly two NUMERIC predictors, so build a separate
	// two-predictor model (bill measurements alone) just for visualization.
	billOnly := datamunge.Must(datamunge.NewNaiveBayesClassifier(penguins, "species ~ bill_length_mm + bill_depth_mm"))
	defer billOnly.Close()
	fmt.Printf("\nbill-measurements-only model training accuracy: %v%%\n", 100*datamunge.Must(billOnly.TrainingAccuracy()))
	check(datamunge.Must(billOnly.PlotClassification(penguins, "bill_length_mm", "bill_depth_mm")).
		Save("naive_bayes_penguins_classification.svg"))
	check(datamunge.Must(billOnly.PlotDecisionRegions("bill_length_mm", "bill_depth_mm")).
		Save("naive_bayes_penguins_decision_regions.svg"))
	fmt.Println("Saved naive_bayes_penguins_classification.svg and" +
		" naive_bayes_penguins_decision_regions.svg")
}
