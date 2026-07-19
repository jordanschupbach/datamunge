package main

import (
	"datamunge"
	"fmt"
	"sort"
)

func dvector(values []float64) datamunge.DVector {
	out := datamunge.NewDVector(int64(len(values)))
	for i, v := range values {
		out.Set(i, v)
	}
	return out
}

func szvector(values []int64) datamunge.SizeVector {
	out := datamunge.NewSizeVector(int64(len(values)))
	for i, v := range values {
		out.Set(i, v)
	}
	return out
}

func printResult(label string, r datamunge.HypothesisTestResult) {
	line := fmt.Sprintf("%s: statistic=%g", label, r.GetStatistic())
	if r.GetParameter1() != 0.0 {
		line += fmt.Sprintf(", df1=%g", r.GetParameter1())
	}
	if r.GetParameter2() != 0.0 {
		line += fmt.Sprintf(", df2=%g", r.GetParameter2())
	}
	altNames := []string{"two.sided", "less", "greater"}
	line += fmt.Sprintf(", p=%g (%s)", r.GetP_value(), altNames[int(r.GetAlternative())])
	if r.GetHas_conf_int() {
		line += fmt.Sprintf(", CI=[%g, %g]", r.GetConf_int_lower(), r.GetConf_int_upper())
	}
	line += fmt.Sprintf(" -- %s", r.GetMethod())
	fmt.Println(line)
}

func columnForSpecies(df datamunge.DataFrame, column string, species string) []float64 {
	out := []float64{}
	n := df.Nrows()
	for i := int64(0); i < n; i++ {
		if df.String_at("Species", i) == species {
			out = append(out, df.Numeric_at(column, i))
		}
	}
	return out
}

func main() {
	iris := datamunge.DataFrameIris()

	setosaPetal := columnForSpecies(iris, "Petal.Length", "setosa")
	versicolorPetal := columnForSpecies(iris, "Petal.Length", "versicolor")
	virginicaPetal := columnForSpecies(iris, "Petal.Length", "virginica")

	fmt.Println("=================== t-tests: petal length, setosa vs. versicolor ===================")
	printResult("Welch two-sample t-test", datamunge.T_test_two_sample(dvector(setosaPetal), dvector(versicolorPetal)))
	printResult("Wilcoxon rank-sum test", datamunge.Wilcoxon_rank_sum_test(dvector(setosaPetal), dvector(versicolorPetal)))

	fmt.Println("\n=================== One-way ANOVA / Kruskal-Wallis across all 3 species ===================")
	allPetal := append(append(append([]float64{}, setosaPetal...), versicolorPetal...), virginicaPetal...)
	sizes := szvector([]int64{int64(len(setosaPetal)), int64(len(versicolorPetal)), int64(len(virginicaPetal))})
	printResult("One-way ANOVA", datamunge.One_way_anova(dvector(allPetal), sizes))
	printResult("Kruskal-Wallis", datamunge.Kruskal_wallis_test(dvector(allPetal), sizes))

	fmt.Println("\n=================== Correlation: sepal length vs. petal length ===================")
	n := iris.Nrows()
	sepalLength := []float64{}
	petalLength := []float64{}
	for i := int64(0); i < n; i++ {
		sepalLength = append(sepalLength, iris.Numeric_at("Sepal.Length", i))
		petalLength = append(petalLength, iris.Numeric_at("Petal.Length", i))
	}
	printResult("Pearson correlation", datamunge.Pearson_correlation_test(dvector(sepalLength), dvector(petalLength)))
	printResult("Spearman correlation", datamunge.Spearman_correlation_test(dvector(sepalLength), dvector(petalLength)))

	fmt.Println("\n=================== F-test: petal length variance, setosa vs. virginica ===================")
	printResult("F test", datamunge.F_test_variance(dvector(setosaPetal), dvector(virginicaPetal)))

	fmt.Println("\n=================== Normality: is sepal length normally distributed within setosa? ===================")
	setosaSepal := columnForSpecies(iris, "Sepal.Length", "setosa")
	printResult("Shapiro-Francia", datamunge.Shapiro_francia_test(dvector(setosaSepal)))
	printResult("KS vs. fitted normal", datamunge.Ks_test_one_sample_normal(dvector(setosaSepal), 5.006, 0.3525))

	fmt.Println("\n=================== Chi-squared / Fisher: is petal length \"long\" independent of species? ===================")
	sortedAll := append([]float64{}, allPetal...)
	sort.Float64s(sortedAll)
	medianAll := sortedAll[len(allPetal)/2]
	setosaLong, setosaShort := countAboveBelow(setosaPetal, medianAll)
	versicolorLong, versicolorShort := countAboveBelow(versicolorPetal, medianAll)
	fmt.Printf("table: setosa=[%d,%d] versicolor=[%d,%d]\n", setosaLong, setosaShort, versicolorLong, versicolorShort)
	tableVec := dvector([]float64{float64(setosaLong), float64(setosaShort), float64(versicolorLong), float64(versicolorShort)})
	printResult("Chi-squared independence", datamunge.Chi_squared_test_independence(tableVec, int64(2), int64(2)))
	printResult("Fisher's exact test", datamunge.Fisher_exact_test_2x2(int64(setosaLong), int64(setosaShort), int64(versicolorLong), int64(versicolorShort)))

	fmt.Println("\n=================== Proportion / binomial: fraction of \"long\" petals overall ===================")
	longCount, _ := countAboveBelow(allPetal, medianAll)
	printResult("One-sample proportion test (vs 0.5)", datamunge.Proportion_test_one_sample(int64(longCount), int64(len(allPetal)), 0.5))
	printResult("Exact binomial test (vs 0.5)", datamunge.Binomial_test(int64(longCount), int64(len(allPetal)), 0.5))
}

func countAboveBelow(values []float64, threshold float64) (int, int) {
	above := 0
	for _, v := range values {
		if v > threshold {
			above++
		}
	}
	return above, len(values) - above
}
