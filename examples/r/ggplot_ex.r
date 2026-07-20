# Demonstrates GGPlot -- the ggplot2-style grammar-of-graphics library.
# Uses the flat ClassName_method(obj, ...) call form throughout, not obj$method(...), per
# memory/datamunge_r_dollar_dispatch_bug.md. Every mutator/void call below is wrapped in
# invisible() -- GGPlot's chainable methods return the mutated object, and Rscript auto-prints
# any un-assigned top-level expression's result, including NULL from save_svg().
library(datamunger)

rgb <- function(r, g, b) {
  color <- RGB()
  RGB_r_set(color, r)
  RGB_g_set(color, g)
  RGB_b_set(color, b)
  color
}

iris <- DataFrame_iris()

# ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Species)) + geom_point()
scatter <- GGPlot(iris, "Sepal.Length", "Sepal.Width", "Species")
invisible(GGPlot_geom_point(scatter))
invisible(GGPlot_labs(scatter, "Iris Sepal Dimensions", "Sepal Length", "Sepal Width"))
invisible(GGPlot_theme_minimal(scatter))
invisible(GGPlot_save_svg(scatter, "r_ggplot_point.svg"))

# + geom_smooth(): an lm() fit line per group, reusing stats::LM internally.
smooth <- GGPlot(iris, "Sepal.Length", "Petal.Length")
invisible(GGPlot_geom_point(smooth, rgb(156, 163, 175), 2.5))
invisible(GGPlot_geom_smooth(smooth))
invisible(GGPlot_labs(smooth, "Petal Length vs Sepal Length With a Linear Fit", "Sepal Length", "Petal Length"))
invisible(GGPlot_save_svg(smooth, "r_ggplot_smooth.svg"))

# geom_bar(): counts a discrete column (stat = "count").
bar <- GGPlot(iris, "Species")
invisible(GGPlot_geom_bar(bar))
invisible(GGPlot_labs(bar, "Observations per Species", "Species", "Count"))
invisible(GGPlot_theme_bw(bar))
invisible(GGPlot_save_svg(bar, "r_ggplot_bar.svg"))

# geom_boxplot(): grouped by a discrete x column.
box <- GGPlot(iris, "Species", "Petal.Width")
invisible(GGPlot_geom_boxplot(box))
invisible(GGPlot_labs(box, "Petal Width by Species", "Species", "Petal Width"))
invisible(GGPlot_save_svg(box, "r_ggplot_boxplot.svg"))

# geom_histogram() + geom_density(): distribution of a single numeric column.
hist_plot <- GGPlot(iris, "Sepal.Length")
invisible(GGPlot_geom_histogram(hist_plot, 20L))
invisible(GGPlot_labs(hist_plot, "Distribution of Sepal Length", "Sepal Length", "Count"))
invisible(GGPlot_save_svg(hist_plot, "r_ggplot_histogram.svg"))

density <- GGPlot(iris, "Sepal.Length")
invisible(GGPlot_geom_density(density))
invisible(GGPlot_labs(density, "Density of Sepal Length", "Sepal Length", "Density"))
invisible(GGPlot_save_svg(density, "r_ggplot_density.svg"))

# facet_wrap(): one panel per Species, composed via RLayout under the hood.
faceted <- GGPlot(iris, "Petal.Length", "Petal.Width")
invisible(GGPlot_geom_point(faceted))
invisible(GGPlot_facet_wrap(faceted, "Species"))
invisible(GGPlot_labs(faceted, "Petal Dimensions", "Petal Length", "Petal Width"))
invisible(GGPlot_save_svg(faceted, "r_ggplot_facet.svg"))

# theme_classic(): no gridlines, dark axes.
classic <- GGPlot(iris, "Sepal.Length", "Sepal.Width", "Species")
invisible(GGPlot_geom_point(classic))
invisible(GGPlot_theme_classic(classic))
invisible(GGPlot_save_svg(classic, "r_ggplot_classic_theme.svg"))

# Note: scale_color_manual() is skipped here -- it takes a std::vector<RGB>, and unlike the
# plain numeric/character/list arguments used above (which SWIG's R backend auto-converts),
# a vector of a wrapped struct type needs its own %template(RGBVector) in datamunger.i plus
# push_back-based construction (the same pattern documented for Lua/Octave in memory/
# datamunge_lua_bindings.md and datamunge_octave_bindings.md) -- not wired up for R.

cat("Wrote 7 SVGs as r_ggplot_*.svg\n")
