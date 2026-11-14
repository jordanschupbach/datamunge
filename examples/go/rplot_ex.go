// CMD: just TARGET=rplot_ex run-go

//go:build ignore

// R base-graphics-style plotting with RPlot / RLayout: plot, hist, barplot, boxplot, pie, qqnorm, layouts.
package main

import (
	"datamunge"
	"fmt"
)

func check(err error) {
	if err != nil {
		panic(err)
	}
}

func main() {
	// plot(x, y, type = "p") then abline() layered on afterward.
	scatter := datamunge.Must(datamunge.RPlotPlotWithLabel(
		[]float64{1.0, 2.0, 3.0, 4.0, 5.0},
		[]float64{2.1, 3.9, 6.2, 7.8, 10.1},
		"p",
		"observed"))
	defer scatter.Close()
	red := datamunge.Must(datamunge.NewRGBRedGreenBlueOpacity(220, 38, 38))
	defer red.Close()
	datamunge.Must(scatter.AblineWithStrokeWidth(0.0, 2.0, red, 1.5))
	datamunge.Must(scatter.Title("plot() + abline()"))
	datamunge.Must(scatter.XLabel("x"))
	datamunge.Must(scatter.YLabel("y"))
	check(scatter.SaveSvg("go_rplot_scatter.svg"))

	// hist(): equal-width binning over the data range.
	histogram := datamunge.Must(datamunge.RPlotHistWithLabel(
		[]float64{1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 7}, 6, "counts"))
	defer histogram.Close()
	check(histogram.SaveSvg("go_rplot_hist.svg"))

	// barplot(): categorical positions with x tick labels.
	barplot := datamunge.Must(datamunge.RPlotBarplotWithNames([]float64{23.0, 41.0, 12.0}, []string{"A", "B", "C"}))
	check(datamunge.Must(barplot.Title("barplot()")).SaveSvg("go_rplot_barplot.svg"))

	// boxplot(): Tukey five-number summary per group.
	groups := [][]float64{
		{2.0, 4, 4, 4, 5, 5, 7, 9},
		{1.0, 2, 2, 2, 3, 3, 3, 3, 4, 20},
	}
	boxplot := datamunge.Must(datamunge.RPlotBoxplotWithNames(groups, []string{"low variance", "has outlier"}))
	check(datamunge.Must(boxplot.Title("boxplot()")).SaveSvg("go_rplot_boxplot.svg"))

	// pie(): wedge areas proportional to value, axes hidden automatically.
	pie := datamunge.Must(datamunge.RPlotPieWithNames([]float64{35.0, 25.0, 20.0, 20.0}, []string{"Q1", "Q2", "Q3", "Q4"}))
	check(datamunge.Must(pie.Title("pie()")).SaveSvg("go_rplot_pie.svg"))

	// curve() is omitted to stay in step with the Lua example (Lua cannot supply the callback).

	// qqnorm() + qqline(): standard-normal Q-Q plot with a fitted reference line.
	residuals := []float64{-2.1, -1.3, -0.8, -0.4, -0.1, 0.2, 0.5, 0.9, 1.4, 2.3}
	qq := datamunge.Must(datamunge.RPlotQqnorm(residuals))
	check(datamunge.Must(qq.Qqline(residuals)).SaveSvg("go_rplot_qqnorm.svg"))

	// par(mfrow = c(1, 2))-style multi-panel composition via RLayout.
	layout := datamunge.Must(datamunge.RLayoutCreate(1, 2))
	defer layout.Close()
	datamunge.Must(layout.Add(scatter))
	datamunge.Must(layout.Add(histogram))
	check(layout.SaveSvg("go_rplot_layout.svg"))

	fmt.Println("Wrote 7 SVGs to go_rplot_*.svg")
}
