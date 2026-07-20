# Demonstrates RPlot / RLayout -- the R-base-graphics-style plotting library.
# Uses the flat ClassName_method(obj, ...) call form throughout, not obj$method(...), per
# memory/datamunge_r_dollar_dispatch_bug.md. curve() is skipped here: it takes a director-based
# Callback, and this package's R bindings have no director support (see the same memory file).
# Every mutator/void call below is wrapped in invisible() -- RPlot's chainable methods return
# the mutated object (mirroring the underlying C++ Plot&/RPlot& chaining), and Rscript
# auto-prints any un-assigned top-level expression's result, including NULL from save_svg().
library(datamunger)

rgb <- function(r, g, b) {
  color <- RGB()
  RGB_r_set(color, r)
  RGB_g_set(color, g)
  RGB_b_set(color, b)
  color
}

# plot(x, y, type = "p") then abline() layered on afterward.
scatter <- RPlot_plot(c(1.0, 2.0, 3.0, 4.0, 5.0), c(2.1, 3.9, 6.2, 7.8, 10.1), "p", "observed")
invisible(RPlot_abline(scatter, 0.0, 2.0, rgb(220, 38, 38), 1.5))
invisible(Plot_title(scatter, "plot() + abline()"))
invisible(Plot_x_label(scatter, "x"))
invisible(Plot_y_label(scatter, "y"))
invisible(Plot_save_svg(scatter, "r_rplot_scatter.svg"))

# hist(): equal-width binning over the data range.
samples <- c(1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 7)
hist_plot <- RPlot_hist(samples, 6, "counts")
invisible(Plot_save_svg(hist_plot, "r_rplot_hist.svg"))

# barplot(): categorical positions with x_tick_labels.
bars <- RPlot_barplot(c(23.0, 41.0, 12.0), c("A", "B", "C"))
invisible(Plot_title(bars, "barplot()"))
invisible(Plot_save_svg(bars, "r_rplot_barplot.svg"))

# boxplot(): Tukey five-number summary per group.
groups <- list(c(2, 4, 4, 4, 5, 5, 7, 9), c(1, 2, 2, 2, 3, 3, 3, 3, 4, 20))
box <- RPlot_boxplot(groups, c("low variance", "has outlier"))
invisible(Plot_title(box, "boxplot()"))
invisible(Plot_save_svg(box, "r_rplot_boxplot.svg"))

# pie(): wedge areas proportional to value, axes hidden automatically.
pie <- RPlot_pie(c(35.0, 25.0, 20.0, 20.0), c("Q1", "Q2", "Q3", "Q4"))
invisible(Plot_title(pie, "pie()"))
invisible(Plot_save_svg(pie, "r_rplot_pie.svg"))

# qqnorm() + qqline(): standard-normal Q-Q plot with a fitted reference line.
residuals <- c(-2.1, -1.3, -0.8, -0.4, -0.1, 0.2, 0.5, 0.9, 1.4, 2.3)
qq <- RPlot_qqnorm(residuals)
invisible(RPlot_qqline(qq, residuals))
invisible(Plot_save_svg(qq, "r_rplot_qqnorm.svg"))

# par(mfrow = c(1, 2))-style multi-panel composition via RLayout.
layout <- RLayout_create(1, 2)
invisible(RLayout_add(layout, scatter))
invisible(RLayout_add(layout, hist_plot))
invisible(RLayout_save_svg(layout, "r_rplot_layout.svg"))

cat("Wrote 7 SVGs as r_rplot_*.svg\n")
