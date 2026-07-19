package main

import (
	"datamunge"
	"fmt"
)

func main() {
	FORMULA := "Petal.Length ~ Petal.Width"

	iris := datamunge.DataFrameIris()
	fmt.Printf("iris: %d rows x %d cols\n", iris.Nrows(), iris.Ncols())
	fmt.Printf("formula: %s\n\n", FORMULA)

	model := datamunge.NewKernelRegression(iris, FORMULA)
	model.Print_summary()

	model.Plot_fit(iris).Save("kernel_regression_iris_fit.svg")
	model.Plot_cv_curve().Save("kernel_regression_iris_cv.svg")
	fmt.Println("\nSaved kernel_regression_iris_fit.svg and kernel_regression_iris_cv.svg")

	small := datamunge.NewKernelRegression(iris, FORMULA, "gaussian", 0.05)
	fmt.Printf("\nbandwidth=0.05 (too small): LOO R-squared=%g  LOO RMSE=%g\n", small.R_squared(), small.Rmse())
	small.Plot_fit(iris).Save("kernel_regression_iris_fit_small_bandwidth.svg")

	large := datamunge.NewKernelRegression(iris, FORMULA, "gaussian", 5.0)
	fmt.Printf("bandwidth=5.0 (too large):  LOO R-squared=%g  LOO RMSE=%g\n", large.R_squared(), large.Rmse())
	large.Plot_fit(iris).Save("kernel_regression_iris_fit_large_bandwidth.svg")

	fmt.Printf("bandwidth=%g (CV-selected): LOO R-squared=%g  LOO RMSE=%g\n", model.Bandwidth(), model.R_squared(), model.Rmse())
	fmt.Println("\nSaved kernel_regression_iris_fit_small_bandwidth.svg and kernel_regression_iris_fit_large_bandwidth.svg")
}
