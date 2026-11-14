// CMD: just TARGET=ggplot_ex run-go

//go:build ignore

// Demonstrates GGPlot -- the ggplot2-style grammar-of-graphics library.
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
	iris := datamunge.Must(data.Iris())
	defer iris.Close()

	// Each GGPlot method returns the plot, so a ggplot2 "+ layer" chain becomes a method chain.
	// ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Species)) + geom_point()
	point := datamunge.Must(datamunge.NewGGPlotWithColorColumn(iris, "Sepal.Length", "Sepal.Width", "Species"))
	datamunge.Must(point.GeomPoint())
	datamunge.Must(point.LabsWithY("Iris Sepal Dimensions", "Sepal Length", "Sepal Width"))
	datamunge.Must(point.ThemeMinimal())
	check(point.SaveSvg("go_ggplot_point.svg"))

	// + geom_smooth(): an lm() fit line, reusing stats::LM internally.
	gray := datamunge.Must(datamunge.NewRGBRedGreenBlueOpacity(156, 163, 175))
	defer gray.Close()
	smooth := datamunge.Must(datamunge.NewGGPlotWithYColumn(iris, "Sepal.Length", "Petal.Length"))
	datamunge.Must(smooth.GeomPointWithSize(gray, 2.5))
	datamunge.Must(smooth.GeomSmooth())
	datamunge.Must(smooth.LabsWithY("Petal Length vs Sepal Length With a Linear Fit", "Sepal Length", "Petal Length"))
	check(smooth.SaveSvg("go_ggplot_smooth.svg"))

	// geom_bar(): counts a discrete column (stat = "count").
	bar := datamunge.Must(datamunge.NewGGPlot(iris, "Species"))
	datamunge.Must(bar.GeomBar())
	datamunge.Must(bar.LabsWithY("Observations per Species", "Species", "Count"))
	datamunge.Must(bar.ThemeBw())
	check(bar.SaveSvg("go_ggplot_bar.svg"))

	// geom_boxplot(): grouped by a discrete x column.
	box := datamunge.Must(datamunge.NewGGPlotWithYColumn(iris, "Species", "Petal.Width"))
	datamunge.Must(box.GeomBoxplot())
	datamunge.Must(box.LabsWithY("Petal Width by Species", "Species", "Petal Width"))
	check(box.SaveSvg("go_ggplot_boxplot.svg"))

	// geom_histogram() + geom_density(): distribution of a single numeric column.
	hist := datamunge.Must(datamunge.NewGGPlot(iris, "Sepal.Length"))
	datamunge.Must(hist.GeomHistogramWithBins(20))
	datamunge.Must(hist.LabsWithY("Distribution of Sepal Length", "Sepal Length", "Count"))
	check(hist.SaveSvg("go_ggplot_histogram.svg"))

	density := datamunge.Must(datamunge.NewGGPlot(iris, "Sepal.Length"))
	datamunge.Must(density.GeomDensity())
	datamunge.Must(density.LabsWithY("Density of Sepal Length", "Sepal Length", "Density"))
	check(density.SaveSvg("go_ggplot_density.svg"))

	// facet_wrap(): one panel per Species, composed via RLayout under the hood.
	facet := datamunge.Must(datamunge.NewGGPlotWithYColumn(iris, "Petal.Length", "Petal.Width"))
	datamunge.Must(facet.GeomPoint())
	datamunge.Must(facet.FacetWrap("Species"))
	datamunge.Must(facet.LabsWithY("Petal Dimensions", "Petal Length", "Petal Width"))
	check(facet.SaveSvg("go_ggplot_facet.svg"))

	// theme_classic(): another built-in theme.
	classic := datamunge.Must(datamunge.NewGGPlotWithColorColumn(iris, "Sepal.Length", "Sepal.Width", "Species"))
	datamunge.Must(classic.GeomPoint())
	datamunge.Must(classic.ThemeClassic())
	check(classic.SaveSvg("go_ggplot_classic.svg"))

	fmt.Println("Wrote 7 SVGs to go_ggplot_*.svg")
}
