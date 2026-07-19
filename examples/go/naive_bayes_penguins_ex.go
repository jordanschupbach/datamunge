package main

import (
	"datamunge"
	"fmt"
)

func main() {
	penguins := datamunge.DataFramePenguins()
	fmt.Printf("penguins: %d rows x %d cols\n\n", penguins.Nrows(), penguins.Ncols())

	model := datamunge.NewNaiveBayesClassifier(penguins, "species ~ bill_length_mm + bill_depth_mm + island + sex")
	model.Print_summary()

	fmt.Println("\nConfusion matrix (rows = actual, cols = predicted):")
	fmt.Println(model.Confusion_matrix().To_string())

	fmt.Println("\nMisclassified rows:")
	predictions := model.Predict(penguins)
	misclassified := 0
	n := penguins.Nrows()
	for i := int64(0); i < n; i++ {
		if !penguins.Is_null("species", i) && !penguins.Is_null("bill_length_mm", i) &&
			!penguins.Is_null("bill_depth_mm", i) && !penguins.Is_null("island", i) && !penguins.Is_null("sex", i) {
			actual := penguins.String_at("species", i)
			pred := predictions.Get(int(i))
			if pred != actual {
				misclassified++
				fmt.Printf("  row %d: bill_length=%g bill_depth=%g island=%s sex=%s  actual=%s  predicted=%s\n", i,
					penguins.Numeric_at("bill_length_mm", i), penguins.Numeric_at("bill_depth_mm", i),
					penguins.String_at("island", i), penguins.String_at("sex", i), actual, pred)
			}
		}
	}
	fmt.Printf("%d misclassified (of %d rows, some incomplete)\n", misclassified, n)

	billOnly := datamunge.NewNaiveBayesClassifier(penguins, "species ~ bill_length_mm + bill_depth_mm")
	fmt.Printf("\nbill-measurements-only model training accuracy: %g%%\n", billOnly.Training_accuracy()*100.0)
	billOnly.Plot_classification(penguins, "bill_length_mm", "bill_depth_mm").Save("naive_bayes_penguins_classification.svg")
	billOnly.Plot_decision_regions("bill_length_mm", "bill_depth_mm").Save("naive_bayes_penguins_decision_regions.svg")
	fmt.Println("Saved naive_bayes_penguins_classification.svg and naive_bayes_penguins_decision_regions.svg")
}
