# CMD: nix develop .#r --command Rscript examples/r/rplot_ex.r
# REPL: nix develop .#r --command R --no-save --quiet
# Demonstrates RPlot / RLayout -- the R-base-graphics-style plotting library.
# Mutator calls below are wrapped in invisible(): RPlot's chainable methods return the plot
# itself (mirroring the underlying C++ Plot&/RPlot& chaining), and Rscript auto-prints any
# un-assigned top-level result.
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

# plot(x, y, type = "p") then abline() layered on afterward.
scatter <- RPlot$plot(c(1.0, 2.0, 3.0, 4.0, 5.0), c(2.1, 3.9, 6.2, 7.8, 10.1), "p", "observed")
invisible(scatter$abline(0.0, 2.0, rgb(220, 38, 38), 1.5))
invisible(scatter$title("plot() + abline()"))
invisible(scatter$x_label("x"))
invisible(scatter$y_label("y"))
invisible(show(scatter))
invisible(scatter$save_svg("r_rplot_scatter.svg"))

# hist(): equal-width binning over the data range.
samples <- c(1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 7)
hist_plot <- RPlot$hist(samples, 6, "counts")
invisible(show(hist_plot))
invisible(hist_plot$save_svg("r_rplot_hist.svg"))

# barplot(): categorical positions with x_tick_labels.
bars <- RPlot$barplot(c(23.0, 41.0, 12.0), c("A", "B", "C"))
invisible(bars$title("barplot()"))
invisible(show(bars))
invisible(bars$save_svg("r_rplot_barplot.svg"))

# boxplot(): Tukey five-number summary per group.
groups <- list(c(2, 4, 4, 4, 5, 5, 7, 9), c(1, 2, 2, 2, 3, 3, 3, 3, 4, 20))
box <- RPlot$boxplot(groups, c("low variance", "has outlier"))
invisible(box$title("boxplot()"))
invisible(show(box))
invisible(box$save_svg("r_rplot_boxplot.svg"))

# pie(): wedge areas proportional to value, axes hidden automatically.
pie <- RPlot$pie(c(35.0, 25.0, 20.0, 20.0), c("Q1", "Q2", "Q3", "Q4"))
invisible(pie$title("pie()"))
invisible(show(pie))
invisible(pie$save_svg("r_rplot_pie.svg"))

# qqnorm() + qqline(): standard-normal Q-Q plot with a fitted reference line.
residuals <- c(-2.1, -1.3, -0.8, -0.4, -0.1, 0.2, 0.5, 0.9, 1.4, 2.3)
qq <- RPlot$qqnorm(residuals)
invisible(qq$qqline(residuals))
invisible(show(qq))
invisible(qq$save_svg("r_rplot_qqnorm.svg"))

# curve(): samples a Callback -- here an R6 subclass -- over a range.
SineWave <- R6::R6Class("SineWave", inherit = Callback, public = list(call = function(x) sin(x)))
sine_curve <- RPlot$curve(SineWave$new(), 0.0, 2.0 * pi, 200L, "sin(x)")
invisible(sine_curve$title("curve()"))
invisible(sine_curve$save_svg("r_rplot_curve.svg"))

# par(mfrow = c(1, 2))-style multi-panel composition via RLayout.
layout <- RLayout$create(1, 2)
invisible(layout$add(scatter))
invisible(layout$add(hist_plot))
invisible(layout$save_svg("r_rplot_layout.svg"))

cat("Wrote 8 SVGs as r_rplot_*.svg\n")
