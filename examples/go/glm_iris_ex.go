package main

import (
	"datamunge"
	"fmt"
	"math"
)

func dvector(values []float64) datamunge.DVector {
	out := datamunge.NewDVector(int64(len(values)))
	for i, v := range values {
		out.Set(i, v)
	}
	return out
}

func main() {
	iris := datamunge.DataFrameIris()

	isVirginica := []float64{}
	petalLength := []float64{}
	petalWidth := []float64{}
	n := iris.Nrows()
	for i := int64(0); i < n; i++ {
		species := iris.String_at("Species", i)
		if species == "versicolor" || species == "virginica" {
			if species == "virginica" {
				isVirginica = append(isVirginica, 1.0)
			} else {
				isVirginica = append(isVirginica, 0.0)
			}
			petalLength = append(petalLength, iris.Numeric_at("Petal.Length", i))
			petalWidth = append(petalWidth, iris.Numeric_at("Petal.Width", i))
		}
	}

	sub := datamunge.NewDataFrame()
	sub.Add_numeric_column("Petal.Length", dvector(petalLength))
	sub.Add_numeric_column("Petal.Width", dvector(petalWidth))
	sub.Add_numeric_column("is_virginica", dvector(isVirginica))

	fmt.Println("=================== Logistic regression (binomial, logit link) ===================")
	logit := datamunge.NewGLM(sub, "is_virginica ~ Petal.Length + Petal.Width", "binomial")
	logit.Print_summary()

	fitted := logit.Fitted_values()
	correct := 0
	nn := len(isVirginica)
	for i := 0; i < nn; i++ {
		if (fitted.Get(i) >= 0.5) == (isVirginica[i] >= 0.5) {
			correct++
		}
	}
	fmt.Printf("\nResubstitution accuracy at 0.5 threshold: %g%%\n", 100.0*float64(correct)/float64(nn))

	logit.Save_diagnostic_plots("glm_logistic_iris")
	fmt.Println("\nSaved glm_logistic_iris_{residuals_vs_fitted,normal_qq,scale_location,residuals_vs_leverage}.svg")

	widthSum := 0.0
	for _, w := range petalWidth {
		widthSum += w
	}
	widthMean := widthSum / float64(nn)
	gridN := 100
	plMin := petalLength[0]
	plMax := petalLength[0]
	for _, v := range petalLength {
		plMin = math.Min(plMin, v)
		plMax = math.Max(plMax, v)
	}
	plMin -= 0.3
	plMax += 0.3
	gridX := make([]float64, gridN)
	gridW := make([]float64, gridN)
	for i := 0; i < gridN; i++ {
		gridX[i] = plMin + (plMax-plMin)*float64(i)/float64(gridN-1)
		gridW[i] = widthMean
	}
	grid := datamunge.NewDataFrame()
	grid.Add_numeric_column("Petal.Length", dvector(gridX))
	grid.Add_numeric_column("Petal.Width", dvector(gridW))
	curveFrame := logit.Predict_frame(grid, "confidence")
	fmt.Println("\nPredicted-probability curve (first 5 rows):")
	fmt.Println(curveFrame.To_string(int64(5)))

	fmt.Println("\n=================== Poisson regression (log link) ===================")
	count := []float64{}
	sepalWidth := []float64{}
	allPetalLength := []float64{}
	for i := int64(0); i < n; i++ {
		count = append(count, math.Round(iris.Numeric_at("Sepal.Length", i)))
		sepalWidth = append(sepalWidth, iris.Numeric_at("Sepal.Width", i))
		allPetalLength = append(allPetalLength, iris.Numeric_at("Petal.Length", i))
	}
	countData := datamunge.NewDataFrame()
	countData.Add_numeric_column("Sepal.Width", dvector(sepalWidth))
	countData.Add_numeric_column("Petal.Length", dvector(allPetalLength))
	countData.Add_numeric_column("count", dvector(count))

	poisson := datamunge.NewGLM(countData, "count ~ Sepal.Width + Petal.Length", "poisson")
	poisson.Print_summary()
}
