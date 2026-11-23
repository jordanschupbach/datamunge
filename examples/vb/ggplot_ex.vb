' CMD: just TARGET=ggplot_ex run-vb
' Demonstrates GGPlot -- the ggplot2-style grammar-of-graphics library.
Option Strict On
Imports System
Imports Datamunge

Module Example
    Sub Main()
        Dim iris = Data.Iris()

        ' ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Species)) + geom_point()
        Call New GGPlot(iris, "Sepal.Length", "Sepal.Width", "Species") _
            .GeomPoint() _
            .Labs("Iris Sepal Dimensions", "Sepal Length", "Sepal Width") _
            .ThemeMinimal() _
            .SaveSvg("vb_ggplot_point.svg")

        ' + geom_smooth(): an lm() fit line, reusing stats::LM internally.
        Call New GGPlot(iris, "Sepal.Length", "Petal.Length") _
            .GeomPoint(New RGB(156, 163, 175), 2.5) _
            .GeomSmooth() _
            .Labs("Petal Length vs Sepal Length With a Linear Fit", "Sepal Length", "Petal Length") _
            .SaveSvg("vb_ggplot_smooth.svg")

        ' geom_bar(): counts a discrete column (stat = "count").
        Call New GGPlot(iris, "Species") _
            .GeomBar() _
            .Labs("Observations per Species", "Species", "Count") _
            .ThemeBw() _
            .SaveSvg("vb_ggplot_bar.svg")

        ' geom_boxplot(): grouped by a discrete x column.
        Call New GGPlot(iris, "Species", "Petal.Width") _
            .GeomBoxplot() _
            .Labs("Petal Width by Species", "Species", "Petal Width") _
            .SaveSvg("vb_ggplot_boxplot.svg")

        ' geom_histogram() + geom_density(): distribution of a single numeric column.
        Call New GGPlot(iris, "Sepal.Length") _
            .GeomHistogram(20UL) _
            .Labs("Distribution of Sepal Length", "Sepal Length", "Count") _
            .SaveSvg("vb_ggplot_histogram.svg")

        Call New GGPlot(iris, "Sepal.Length") _
            .GeomDensity() _
            .Labs("Density of Sepal Length", "Sepal Length", "Density") _
            .SaveSvg("vb_ggplot_density.svg")

        ' facet_wrap(): one panel per Species, composed via RLayout under the hood.
        Call New GGPlot(iris, "Petal.Length", "Petal.Width") _
            .GeomPoint() _
            .FacetWrap("Species") _
            .Labs("Petal Dimensions", "Petal Length", "Petal Width") _
            .SaveSvg("vb_ggplot_facet.svg")

        ' theme_classic(): another built-in theme.
        Call New GGPlot(iris, "Sepal.Length", "Sepal.Width", "Species") _
            .GeomPoint() _
            .ThemeClassic() _
            .SaveSvg("vb_ggplot_classic.svg")

        Console.WriteLine("Wrote 7 SVGs to vb_ggplot_*.svg")
    End Sub
End Module
