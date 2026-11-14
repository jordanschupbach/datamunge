# CMD: just TARGET=ggplot_ex run-julia

# Demonstrates GGPlot -- the ggplot2-style grammar-of-graphics library.
using Datamunge

function rgb(r, g, b)
    c = RGB()
    c.r, c.g, c.b = r, g, b
    c
end

const iris = Datamunge.Data.iris()

# ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Species)) + geom_point()
let scatter = GGPlot(iris, "Sepal.Length", "Sepal.Width", "Species")
    geom_point(scatter)
    labs(scatter, "Iris Sepal Dimensions", "Sepal Length", "Sepal Width")
    theme_minimal(scatter)
    save_svg(scatter, "julia_ggplot_point.svg")
end

# + geom_smooth(): an lm() fit line, reusing stats::LM internally.
let smooth = GGPlot(iris, "Sepal.Length", "Petal.Length")
    geom_point(smooth, rgb(156, 163, 175), 2.5)
    geom_smooth(smooth)
    labs(smooth, "Petal Length vs Sepal Length With a Linear Fit", "Sepal Length", "Petal Length")
    save_svg(smooth, "julia_ggplot_smooth.svg")
end

# geom_bar(): counts a discrete column (stat = "count").
let bar = GGPlot(iris, "Species")
    geom_bar(bar)
    labs(bar, "Observations per Species", "Species", "Count")
    theme_bw(bar)
    save_svg(bar, "julia_ggplot_bar.svg")
end

# geom_boxplot(): grouped by a discrete x column.
let box = GGPlot(iris, "Species", "Petal.Width")
    geom_boxplot(box)
    labs(box, "Petal Width by Species", "Species", "Petal Width")
    save_svg(box, "julia_ggplot_boxplot.svg")
end

# geom_histogram() + geom_density(): distribution of a single numeric column.
let hist = GGPlot(iris, "Sepal.Length")
    geom_histogram(hist, 20)
    labs(hist, "Distribution of Sepal Length", "Sepal Length", "Count")
    save_svg(hist, "julia_ggplot_histogram.svg")
end

let density = GGPlot(iris, "Sepal.Length")
    geom_density(density)
    labs(density, "Density of Sepal Length", "Sepal Length", "Density")
    save_svg(density, "julia_ggplot_density.svg")
end

# facet_wrap(): one panel per Species, composed via RLayout under the hood.
let faceted = GGPlot(iris, "Petal.Length", "Petal.Width")
    geom_point(faceted)
    facet_wrap(faceted, "Species")
    labs(faceted, "Petal Dimensions", "Petal Length", "Petal Width")
    save_svg(faceted, "julia_ggplot_facet.svg")
end

# theme_classic(): another built-in theme.
let classic = GGPlot(iris, "Sepal.Length", "Sepal.Width", "Species")
    geom_point(classic)
    theme_classic(classic)
    save_svg(classic, "julia_ggplot_classic.svg")
end

println("Wrote 7 SVGs to julia_ggplot_*.svg")
