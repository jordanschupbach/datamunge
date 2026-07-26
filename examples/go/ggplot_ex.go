package main

import (
	"datamunge"
	"fmt"
)

func rgb(r, g, b int) datamunge.RGB {
	c := datamunge.NewRGB()
	c.SetR(r)
	c.SetG(g)
	c.SetB(b)
	return c
}

func main() {
	iris := datamunge.DataFrameIris()

	// ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Species)) + geom_point()
	scatter := datamunge.NewGGPlot(iris, "Sepal.Length", "Sepal.Width", "Species")
	scatter.Geom_point()
	scatter.Labs("Iris Sepal Dimensions", "Sepal Length", "Sepal Width")
	scatter.Theme_minimal()
	scatter.Save_svg("go_ggplot_point.svg")

	// + geom_smooth(): an lm() fit line, reusing stats::LM internally.
	smooth := datamunge.NewGGPlot(iris, "Sepal.Length", "Petal.Length")
	smooth.Geom_point(rgb(156, 163, 175), 2.5)
	smooth.Geom_smooth()
	smooth.Labs("Petal Length vs Sepal Length With a Linear Fit", "Sepal Length", "Petal Length")
	smooth.Save_svg("go_ggplot_smooth.svg")

	// geom_bar(): counts a discrete column (stat = "count").
	bar := datamunge.NewGGPlot(iris, "Species")
	bar.Geom_bar()
	bar.Labs("Observations per Species", "Species", "Count")
	bar.Theme_bw()
	bar.Save_svg("go_ggplot_bar.svg")

	// geom_boxplot(): grouped by a discrete x column.
	box := datamunge.NewGGPlot(iris, "Species", "Petal.Width")
	box.Geom_boxplot()
	box.Labs("Petal Width by Species", "Species", "Petal Width")
	box.Save_svg("go_ggplot_boxplot.svg")

	// geom_histogram() + geom_density(): distribution of a single numeric column.
	hist := datamunge.NewGGPlot(iris, "Sepal.Length")
	hist.Geom_histogram(int64(20))
	hist.Labs("Distribution of Sepal Length", "Sepal Length", "Count")
	hist.Save_svg("go_ggplot_histogram.svg")

	density := datamunge.NewGGPlot(iris, "Sepal.Length")
	density.Geom_density()
	density.Labs("Density of Sepal Length", "Sepal Length", "Density")
	density.Save_svg("go_ggplot_density.svg")

	// facet_wrap(): one panel per Species, composed via RLayout under the hood.
	faceted := datamunge.NewGGPlot(iris, "Petal.Length", "Petal.Width")
	faceted.Geom_point()
	faceted.Facet_wrap("Species")
	faceted.Labs("Petal Dimensions", "Petal Length", "Petal Width")
	faceted.Save_svg("go_ggplot_facet.svg")

	// theme_classic(): another built-in theme. (scale_color_manual is omitted here -- the
	// std::vector<RGB> overload it needs is not part of the Go binding's convenient surface.)
	classic := datamunge.NewGGPlot(iris, "Sepal.Length", "Sepal.Width", "Species")
	classic.Geom_point()
	classic.Theme_classic()
	classic.Save_svg("go_ggplot_classic.svg")

	fmt.Println("Wrote 7 SVGs to go_ggplot_*.svg")
}
