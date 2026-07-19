package main

import (
	"datamunge"
	"fmt"
)

func main() {
	FORMULA := "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width + Sepal.Length:Petal.Width"

	iris := datamunge.DataFrameIris()
	fmt.Printf("iris: %d rows x %d cols\n", iris.Nrows(), iris.Ncols())
	fmt.Printf("formula: %s\n\n", FORMULA)

	fmt.Println("=================== Ridge ===================")
	ridge := datamunge.NewRidge(iris, FORMULA)
	ridge.Print_summary()

	fmt.Println("\n=================== Lasso ===================")
	lasso := datamunge.NewLasso(iris, FORMULA)
	lasso.Print_summary()

	fmt.Println("\n================= Elastic Net =================")
	elastic := datamunge.NewElasticNet(iris, FORMULA, 0.5)
	elastic.Print_summary()

	fmt.Println("\nSaved figures showing how each model's coefficients respond to the regularization strength, and the cross-validation curve used to pick it:")

	ridge.Plot_coefficient_path().Save("elastic_net_ridge_path.svg")
	ridge.Plot_cv_curve().Save("elastic_net_ridge_cv.svg")
	fmt.Println("  ridge:       elastic_net_ridge_path.svg, elastic_net_ridge_cv.svg")

	lasso.Plot_coefficient_path().Save("elastic_net_lasso_path.svg")
	lasso.Plot_cv_curve().Save("elastic_net_lasso_cv.svg")
	fmt.Println("  lasso:       elastic_net_lasso_path.svg, elastic_net_lasso_cv.svg")

	elastic.Plot_coefficient_path().Save("elastic_net_elasticnet_path.svg")
	elastic.Plot_cv_curve().Save("elastic_net_elasticnet_cv.svg")
	fmt.Println("  elastic net: elastic_net_elasticnet_path.svg, elastic_net_elasticnet_cv.svg")
}
