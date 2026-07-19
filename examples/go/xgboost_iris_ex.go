package main

import (
	"datamunge"
	"fmt"
)

func main() {
	iris := datamunge.DataFrameIris()
	fmt.Printf("iris: %d rows x %d cols\n\n", iris.Nrows(), iris.Ncols())

	model := datamunge.NewXGBoostClassifier(iris, "Species ~ Petal.Length + Petal.Width")
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

	model.Plot_training_deviance().Save("xgboost_iris_training_deviance.svg")
	model.Plot_decision_regions("Petal.Length", "Petal.Width").Save("xgboost_iris_decision_regions.svg")
	fmt.Println("\nSaved xgboost_iris_training_deviance.svg and xgboost_iris_decision_regions.svg")

	heavy := datamunge.NewXGBoostClassifier(iris, "Species ~ Petal.Length + Petal.Width", int64(100), 0.3, int64(6), 50.0)
	modelDev := model.Training_deviance()
	heavyDev := heavy.Training_deviance()
	fmt.Printf("\nlambda=1 (default):   training accuracy=%g%%  deviance=%g\n", model.Training_accuracy()*100.0, modelDev.Get(int(modelDev.Size())-1))
	fmt.Printf("lambda=50 (heavy L2): training accuracy=%g%%  deviance=%g\n", heavy.Training_accuracy()*100.0, heavyDev.Get(int(heavyDev.Size())-1))
	heavy.Plot_decision_regions("Petal.Length", "Petal.Width").Save("xgboost_iris_decision_regions_heavy_lambda.svg")
	fmt.Println("Saved xgboost_iris_decision_regions_heavy_lambda.svg")
}
