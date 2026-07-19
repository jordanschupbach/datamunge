package main

import (
	"datamunge"
	"fmt"
)

func main() {
	iris := datamunge.DataFrameIris()
	fmt.Printf("iris: %d rows x %d cols\n\n", iris.Nrows(), iris.Ncols())

	model := datamunge.NewGBMClassifier(iris, "Species ~ Petal.Length + Petal.Width")
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

	model.Plot_training_deviance().Save("gbm_iris_training_deviance.svg")
	model.Plot_decision_regions("Petal.Length", "Petal.Width").Save("gbm_iris_decision_regions.svg")
	fmt.Println("\nSaved gbm_iris_training_deviance.svg and gbm_iris_decision_regions.svg")

	few := datamunge.NewGBMClassifier(iris, "Species ~ Petal.Length + Petal.Width", int64(5))
	fewDev := few.Training_deviance()
	modelDev := model.Training_deviance()
	fmt.Printf("\n5-round ensemble:   training accuracy=%g%%  deviance=%g\n", few.Training_accuracy()*100.0, fewDev.Get(int(fewDev.Size())-1))
	fmt.Printf("100-round ensemble: training accuracy=%g%%  deviance=%g\n", model.Training_accuracy()*100.0, modelDev.Get(int(modelDev.Size())-1))
	few.Plot_decision_regions("Petal.Length", "Petal.Width").Save("gbm_iris_decision_regions_5rounds.svg")
	fmt.Println("Saved gbm_iris_decision_regions_5rounds.svg")
}
