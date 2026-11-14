// CMD: just TARGET=elastic_net_iris_ex run-go

//go:build ignore

// Ridge, Lasso and Elastic Net on iris, with coefficient-path and CV-curve figures.
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

func savePlots(name, label string, path, cv *datamunge.RPlot) {
	check(path.Save("elastic_net_" + name + "_path.svg"))
	check(cv.Save("elastic_net_" + name + "_cv.svg"))
	fmt.Printf("  %s elastic_net_%s_path.svg, elastic_net_%s_cv.svg\n", label, name, name)
}

func main() {
	formula := "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width + Sepal.Length:Petal.Width"

	iris := datamunge.Must(data.Iris())
	defer iris.Close()
	fmt.Printf("iris: %d rows x %d cols\n", datamunge.Must(iris.Nrows()), datamunge.Must(iris.Ncols()))
	fmt.Printf("formula: %s\n\n", formula)

	// Summary() returns the same text PrintSummary() writes, printed here through fmt.
	fmt.Println("=================== Ridge ===================")
	ridge := datamunge.Must(datamunge.NewRidge(iris, formula))
	defer ridge.Close()
	fmt.Print(datamunge.Must(ridge.Summary()))

	fmt.Println("\n=================== Lasso ===================")
	lasso := datamunge.Must(datamunge.NewLasso(iris, formula))
	defer lasso.Close()
	fmt.Print(datamunge.Must(lasso.Summary()))

	fmt.Println("\n================= Elastic Net =================")
	// NewElasticNetWith<LastParam>(data, formula, alpha, lambda, nLambda, cvFolds, standardize, seed)
	elastic := datamunge.Must(datamunge.NewElasticNetWithAlpha(iris, formula, 0.5))
	defer elastic.Close()
	fmt.Print(datamunge.Must(elastic.Summary()))

	fmt.Println("\nSaved figures showing how each model's coefficients respond to the regularization" +
		" strength, and the cross-validation curve used to pick it:")

	savePlots("ridge", "ridge:      ", datamunge.Must(ridge.PlotCoefficientPath()), datamunge.Must(ridge.PlotCvCurve()))
	savePlots("lasso", "lasso:      ", datamunge.Must(lasso.PlotCoefficientPath()), datamunge.Must(lasso.PlotCvCurve()))
	savePlots("elasticnet", "elastic net:", datamunge.Must(elastic.PlotCoefficientPath()), datamunge.Must(elastic.PlotCvCurve()))
}
