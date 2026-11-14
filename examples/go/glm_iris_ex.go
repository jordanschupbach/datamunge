// CMD: just TARGET=glm_iris_ex run-go

//go:build ignore

// Generalized linear models on iris: logistic regression (binomial) and Poisson regression.
package main

import (
	"datamunge"
	"datamunge/data"
	"fmt"
	"math"
	"slices"
)

func check(err error) {
	if err != nil {
		panic(err)
	}
}

func column(df *datamunge.DataFrame, name string, rows []int) []float64 {
	out := make([]float64, len(rows))
	for k, i := range rows {
		out[k] = datamunge.Must(df.NumericAt(name, i))
	}
	return out
}

func mean(values []float64) float64 {
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

func main() {
	iris := datamunge.Must(data.Iris())
	defer iris.Close()
	rows := make([]int, datamunge.Must(iris.Nrows()))
	for i := range rows {
		rows[i] = i
	}

	// Logistic regression: versicolor vs virginica only -- setosa is perfectly separable from
	// the other two on these predictors, which sends logistic regression's coefficients toward
	// +/-infinity (a genuine degeneracy of the method, not a bug) -- so this is the well-behaved
	// binary split for a demo.
	kept := []int{}
	isVirginica := []float64{}
	for _, i := range rows {
		sp := datamunge.Must(iris.StringAt("Species", i))
		if sp == "versicolor" || sp == "virginica" {
			kept = append(kept, i)
			if sp == "virginica" {
				isVirginica = append(isVirginica, 1.0)
			} else {
				isVirginica = append(isVirginica, 0.0)
			}
		}
	}
	petalLength := column(iris, "Petal.Length", kept)
	petalWidth := column(iris, "Petal.Width", kept)

	sub := datamunge.Must(datamunge.NewDataFrame())
	defer sub.Close()
	check(sub.AddNumericColumn("Petal.Length", petalLength))
	check(sub.AddNumericColumn("Petal.Width", petalWidth))
	check(sub.AddNumericColumn("is_virginica", isVirginica))

	fmt.Println("=================== Logistic regression (binomial, logit link) ===================")
	logit := datamunge.Must(datamunge.NewGLMWithFamily(sub, "is_virginica ~ Petal.Length + Petal.Width", "binomial"))
	defer logit.Close()
	check(logit.PrintSummary())

	n := len(isVirginica)
	correct := 0
	fitted := datamunge.Must(logit.FittedValues())
	for i := 0; i < n; i++ {
		if (fitted[i] >= 0.5) == (isVirginica[i] >= 0.5) {
			correct++
		}
	}
	fmt.Printf("\nResubstitution accuracy at 0.5 threshold: %v%%\n", 100.0*float64(correct)/float64(n))

	check(logit.SaveDiagnosticPlots("glm_logistic_iris"))
	fmt.Println("\nSaved glm_logistic_iris_{residuals_vs_fitted,normal_qq,scale_location," +
		"residuals_vs_leverage}.svg")

	// Predicted-probability curve across Petal.Length, with Petal.Width held at its mean -- the
	// classic sigmoid shape of a fitted logistic regression, with a 95% confidence band.
	gridN := 100
	widthMean := mean(petalWidth)
	plMin := slices.Min(petalLength) - 0.3
	plMax := slices.Max(petalLength) + 0.3
	gridX := make([]float64, gridN)
	gridWidth := make([]float64, gridN)
	for i := range gridX {
		gridX[i] = plMin + (plMax-plMin)*float64(i)/float64(gridN-1)
		gridWidth[i] = widthMean
	}
	grid := datamunge.Must(datamunge.NewDataFrame())
	defer grid.Close()
	check(grid.AddNumericColumn("Petal.Length", gridX))
	check(grid.AddNumericColumn("Petal.Width", gridWidth))
	curveFrame := datamunge.Must(logit.PredictFrameWithIntervalKind(grid, "confidence"))
	defer curveFrame.Close()
	fmt.Println("\nPredicted-probability curve (first 5 rows):")
	fmt.Println(datamunge.Must(curveFrame.ToStringWithMaxRows(5)))

	// Poisson regression, for contrast: same IRLS engine, different family/link.
	fmt.Println("\n=================== Poisson regression (log link) ===================")
	counts := make([]float64, len(rows))
	for k, i := range rows {
		counts[k] = math.Floor(datamunge.Must(iris.NumericAt("Sepal.Length", i)) + 0.5)
	}
	countData := datamunge.Must(datamunge.NewDataFrame())
	defer countData.Close()
	check(countData.AddNumericColumn("Sepal.Width", column(iris, "Sepal.Width", rows)))
	check(countData.AddNumericColumn("Petal.Length", column(iris, "Petal.Length", rows)))
	check(countData.AddNumericColumn("count", counts))
	poisson := datamunge.Must(datamunge.NewGLMWithFamily(countData, "count ~ Sepal.Width + Petal.Length", "poisson"))
	defer poisson.Close()
	check(poisson.PrintSummary())
}
