// CMD: just TARGET=kernel_regression_iris_ex run-go

//go:build ignore

// Nadaraya-Watson kernel regression on iris: CV-selected bandwidth vs. too-small and too-large ones.
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
	formula := "Petal.Length ~ Petal.Width"

	iris := datamunge.Must(data.Iris())
	defer iris.Close()
	fmt.Printf("iris: %d rows x %d cols\n", datamunge.Must(iris.Nrows()), datamunge.Must(iris.Ncols()))
	fmt.Printf("formula: %s\n\n", formula)

	// Bandwidth auto-selected via leave-one-out CV.
	model := datamunge.Must(datamunge.NewKernelRegression(iris, formula))
	defer model.Close()
	check(model.PrintSummary())

	check(datamunge.Must(model.PlotFit(iris)).Save("kernel_regression_iris_fit.svg"))
	check(datamunge.Must(model.PlotCvCurve()).Save("kernel_regression_iris_cv.svg"))
	fmt.Println("\nSaved kernel_regression_iris_fit.svg and kernel_regression_iris_cv.svg")

	// A too-small and a too-large bandwidth, for comparison -- the classic kernel regression
	// bias-variance story. NewKernelRegressionWithBandwidth(data, formula, kernel, bandwidth)
	small := datamunge.Must(datamunge.NewKernelRegressionWithBandwidth(iris, formula, "gaussian", 0.05))
	defer small.Close()
	fmt.Printf("\nbandwidth=0.05 (too small): LOO R-squared=%v  LOO RMSE=%v\n",
		datamunge.Must(small.RSquared()), datamunge.Must(small.Rmse()))
	check(datamunge.Must(small.PlotFit(iris)).Save("kernel_regression_iris_fit_small_bandwidth.svg"))

	large := datamunge.Must(datamunge.NewKernelRegressionWithBandwidth(iris, formula, "gaussian", 5.0))
	defer large.Close()
	fmt.Printf("bandwidth=5.0 (too large):  LOO R-squared=%v  LOO RMSE=%v\n",
		datamunge.Must(large.RSquared()), datamunge.Must(large.Rmse()))
	check(datamunge.Must(large.PlotFit(iris)).Save("kernel_regression_iris_fit_large_bandwidth.svg"))

	fmt.Printf("bandwidth=%v (CV-selected): LOO R-squared=%v  LOO RMSE=%v\n",
		datamunge.Must(model.Bandwidth()), datamunge.Must(model.RSquared()), datamunge.Must(model.Rmse()))
	fmt.Println("\nSaved kernel_regression_iris_fit_small_bandwidth.svg and" +
		" kernel_regression_iris_fit_large_bandwidth.svg")
}
