# CMD: nix develop .#r --command env R_LIBS_USER=build/r/library Rscript examples/r/ggplot_ex.r
# REPL: nix develop .#r --command env R_LIBS_USER=build/r/library R --no-save --quiet
# Demonstrates GGPlot -- the ggplot2-style grammar-of-graphics library.
# Mutator calls below are wrapped in invisible(): GGPlot's chainable methods return the
# plot itself, and Rscript auto-prints any un-assigned top-level result.
library(datamunger)

# show() opens an X11 window; skip it with a note when there is no display or the library was
# built without X11. The SVGs are still written below.
show <- function(plot) {
  if (!nzchar(Sys.getenv("DISPLAY"))) return(invisible(message("  (show skipped -- no DISPLAY)")))
  tryCatch(plot$show(), error = function(e) message("  (show skipped -- ", conditionMessage(e), ")"))
  invisible(plot)
}

rgb <- function(r, g, b) {
  color <- RGB$new()
  color$r <- r
  color$g <- g
  color$b <- b
  color
}

iris <- data$iris()

# ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Species)) + geom_point()
scatter <- GGPlot$new(iris, "Sepal.Length", "Sepal.Width", "Species")
invisible(scatter$geom_point())
invisible(scatter$labs("Iris Sepal Dimensions", "Sepal Length", "Sepal Width"))
invisible(scatter$theme_minimal())
invisible(show(scatter))
invisible(scatter$save_svg("r_ggplot_point.svg"))

# + geom_smooth(): an lm() fit line per group, reusing stats::LM internally.
smooth <- GGPlot$new(iris, "Sepal.Length", "Petal.Length")
invisible(smooth$geom_point(rgb(156, 163, 175), 2.5))
invisible(smooth$geom_smooth())
invisible(smooth$labs("Petal Length vs Sepal Length With a Linear Fit", "Sepal Length", "Petal Length"))
invisible(show(smooth))
invisible(smooth$save_svg("r_ggplot_smooth.svg"))

# geom_bar(): counts a discrete column (stat = "count").
bar <- GGPlot$new(iris, "Species")
invisible(bar$geom_bar())
invisible(bar$labs("Observations per Species", "Species", "Count"))
invisible(bar$theme_bw())
invisible(show(bar))
invisible(bar$save_svg("r_ggplot_bar.svg"))

# geom_boxplot(): grouped by a discrete x column.
box <- GGPlot$new(iris, "Species", "Petal.Width")
invisible(box$geom_boxplot())
invisible(box$labs("Petal Width by Species", "Species", "Petal Width"))
invisible(show(box))
invisible(box$save_svg("r_ggplot_boxplot.svg"))

# geom_histogram() + geom_density(): distribution of a single numeric column.
hist_plot <- GGPlot$new(iris, "Sepal.Length")
invisible(hist_plot$geom_histogram(20L))
invisible(hist_plot$labs("Distribution of Sepal Length", "Sepal Length", "Count"))
invisible(show(hist_plot))
invisible(hist_plot$save_svg("r_ggplot_histogram.svg"))

density <- GGPlot$new(iris, "Sepal.Length")
invisible(density$geom_density())
invisible(density$labs("Density of Sepal Length", "Sepal Length", "Density"))
invisible(show(density))
invisible(density$save_svg("r_ggplot_density.svg"))

# facet_wrap(): one panel per Species, composed via RLayout under the hood.
faceted <- GGPlot$new(iris, "Petal.Length", "Petal.Width")
invisible(faceted$geom_point())
invisible(faceted$facet_wrap("Species"))
invisible(faceted$labs("Petal Dimensions", "Petal Length", "Petal Width"))
invisible(show(faceted))
invisible(faceted$save_svg("r_ggplot_facet.svg"))

# theme_classic(): no gridlines, dark axes.
classic <- GGPlot$new(iris, "Sepal.Length", "Sepal.Width", "Species")
invisible(classic$geom_point())
invisible(classic$theme_classic())
invisible(show(classic))
invisible(classic$save_svg("r_ggplot_classic_theme.svg"))

# scale_color_manual() takes a std::vector<RGB>, passed from R as a list of RGB objects:
#   plot$scale_color_manual(list(RGB$new("red"), RGB$new("blue")))

cat("Wrote 7 SVGs as r_ggplot_*.svg\n")
