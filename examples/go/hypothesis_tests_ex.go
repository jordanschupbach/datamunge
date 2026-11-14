// CMD: just TARGET=hypothesis_tests_ex run-go

//go:build ignore

// Classical hypothesis tests (t, Wilcoxon, ANOVA, correlation, F, normality, chi-squared, binomial) on iris.
package main

import (
	"datamunge"
	"datamunge/data"
	"fmt"
	"slices"
	"strings"
)

func printResult(label string, r *datamunge.HypothesisTestResult) {
	defer r.Close()
	alternative := map[datamunge.Alternative]string{
		datamunge.AlternativeTwoSided: "two.sided",
		datamunge.AlternativeLess:     "less",
		datamunge.AlternativeGreater:  "greater",
	}[r.Alternative()]
	var line strings.Builder
	fmt.Fprintf(&line, "%s: statistic=%v", label, r.Statistic())
	if r.Parameter1() != 0 {
		fmt.Fprintf(&line, ", df1=%v", r.Parameter1())
	}
	if r.Parameter2() != 0 {
		fmt.Fprintf(&line, ", df2=%v", r.Parameter2())
	}
	fmt.Fprintf(&line, ", p=%v (%s)", r.PValue(), alternative)
	if r.HasConfInt() {
		fmt.Fprintf(&line, ", CI=[%v, %v]", r.ConfIntLower(), r.ConfIntUpper())
	}
	fmt.Println(line.String() + " -- " + r.Method())
}

// Values of a numeric column, optionally only for rows of one species ("" = all rows).
func column(df *datamunge.DataFrame, name, species string) []float64 {
	var values []float64
	n := datamunge.Must(df.Nrows())
	for i := 0; i < n; i++ {
		if species == "" || datamunge.Must(df.StringAt("Species", i)) == species {
			values = append(values, datamunge.Must(df.NumericAt(name, i)))
		}
	}
	return values
}

// Number of values strictly greater than threshold.
func countAbove(values []float64, threshold float64) int {
	count := 0
	for _, x := range values {
		if x > threshold {
			count++
		}
	}
	return count
}

func main() {
	iris := datamunge.Must(data.Iris())
	defer iris.Close()

	setosaPetal := column(iris, "Petal.Length", "setosa")
	versicolorPetal := column(iris, "Petal.Length", "versicolor")
	virginicaPetal := column(iris, "Petal.Length", "virginica")

	fmt.Println("=================== t-tests: petal length, setosa vs. versicolor ===================")
	printResult("Welch two-sample t-test", datamunge.Must(datamunge.TTestTwoSample(setosaPetal, versicolorPetal)))
	printResult("Wilcoxon rank-sum test", datamunge.Must(datamunge.WilcoxonRankSumTest(setosaPetal, versicolorPetal)))

	fmt.Println("\n=================== One-way ANOVA / Kruskal-Wallis across all 3 species ===================")
	allPetal := slices.Concat(setosaPetal, versicolorPetal, virginicaPetal)
	sizes := []int{len(setosaPetal), len(versicolorPetal), len(virginicaPetal)}
	printResult("One-way ANOVA", datamunge.Must(datamunge.OneWayAnova(allPetal, sizes)))
	printResult("Kruskal-Wallis", datamunge.Must(datamunge.KruskalWallisTest(allPetal, sizes)))

	fmt.Println("\n=================== Correlation: sepal length vs. petal length ===================")
	sepalLength := column(iris, "Sepal.Length", "")
	petalLength := column(iris, "Petal.Length", "")
	printResult("Pearson correlation", datamunge.Must(datamunge.PearsonCorrelationTest(sepalLength, petalLength)))
	printResult("Spearman correlation", datamunge.Must(datamunge.SpearmanCorrelationTest(sepalLength, petalLength)))

	fmt.Println("\n=================== F-test: petal length variance, setosa vs. virginica ===================")
	printResult("F test", datamunge.Must(datamunge.FTestVariance(setosaPetal, virginicaPetal)))

	fmt.Println("\n=================== Normality: is sepal length normally distributed within setosa? ===================")
	setosaSepal := column(iris, "Sepal.Length", "setosa")
	printResult("Shapiro-Francia", datamunge.Must(datamunge.ShapiroFranciaTest(setosaSepal)))
	printResult("KS vs. fitted normal", datamunge.Must(datamunge.KsTestOneSampleNormalWithSd(setosaSepal, 5.006, 0.3525)))

	fmt.Println("\n=================== Chi-squared / Fisher: is petal length \"long\" independent of species? ===================")
	sortedPetal := slices.Clone(allPetal)
	slices.Sort(sortedPetal)
	medianAll := sortedPetal[len(sortedPetal)/2]
	setosaLong := countAbove(setosaPetal, medianAll)
	setosaShort := len(setosaPetal) - setosaLong
	versicolorLong := countAbove(versicolorPetal, medianAll)
	versicolorShort := len(versicolorPetal) - versicolorLong
	fmt.Printf("table: setosa=[%d,%d] versicolor=[%d,%d]\n", setosaLong, setosaShort, versicolorLong, versicolorShort)
	table := []float64{float64(setosaLong), float64(setosaShort), float64(versicolorLong), float64(versicolorShort)}
	printResult("Chi-squared independence", datamunge.Must(datamunge.ChiSquaredTestIndependence(table, 2, 2)))
	printResult("Fisher's exact test",
		datamunge.Must(datamunge.FisherExactTest2x2(setosaLong, setosaShort, versicolorLong, versicolorShort)))

	fmt.Println("\n=================== Proportion / binomial: fraction of \"long\" petals overall ===================")
	longCount := countAbove(allPetal, medianAll)
	n := len(allPetal)
	printResult("One-sample proportion test (vs 0.5)", datamunge.Must(datamunge.ProportionTestOneSampleWithP(longCount, n, 0.5)))
	printResult("Exact binomial test (vs 0.5)", datamunge.Must(datamunge.BinomialTestWithP(longCount, n, 0.5)))
}
