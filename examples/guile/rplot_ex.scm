;; CMD: just TARGET=rplot_ex run-guile
;; R base-graphics-style plotting with RPlot / RLayout: plot, hist, barplot, boxplot, pie, qqnorm, layouts.
(use-modules (oop goops) (datamunge))

;; plot(x, y, type = "p") then abline() layered on afterward.
(define scatter (r-plot-plot '(1.0 2.0 3.0 4.0 5.0) '(2.1 3.9 6.2 7.8 10.1) "p" "observed"))
(abline scatter 0.0 2.0 (make <rgb> 220 38 38) 1.5)
(y-label (x-label (title scatter "plot() + abline()") "x") "y")
(save-svg scatter "guile_rplot_scatter.svg")

;; hist(): equal-width binning over the data range.
(define histogram (r-plot-hist '(1.0 2 2 3 3 3 4 4 4 4 5 5 5 6 6 7) 6 "counts"))
(save-svg histogram "guile_rplot_hist.svg")

;; barplot(): categorical positions with x tick labels.
(save-svg (title (r-plot-barplot '(23.0 41.0 12.0) '("A" "B" "C")) "barplot()")
          "guile_rplot_barplot.svg")

;; boxplot(): Tukey five-number summary per group.
(save-svg (title (r-plot-boxplot '((2.0 4 4 4 5 5 7 9) (1.0 2 2 2 3 3 3 3 4 20))
                                 '("low variance" "has outlier"))
                 "boxplot()")
          "guile_rplot_boxplot.svg")

;; pie(): wedge areas proportional to value, axes hidden automatically.
(save-svg (title (r-plot-pie '(35.0 25.0 20.0 20.0) '("Q1" "Q2" "Q3" "Q4")) "pie()")
          "guile_rplot_pie.svg")

;; curve() is omitted to stay in step with the other language examples (Lua cannot supply the callback).

;; qqnorm() + qqline(): standard-normal Q-Q plot with a fitted reference line.
(define residuals '(-2.1 -1.3 -0.8 -0.4 -0.1 0.2 0.5 0.9 1.4 2.3))
(save-svg (qqline (r-plot-qqnorm residuals) residuals) "guile_rplot_qqnorm.svg")

;; par(mfrow = c(1, 2))-style multi-panel composition via RLayout.
(save-svg (add (add (r-layout-create 1 2) scatter) histogram) "guile_rplot_layout.svg")

(display "Wrote 7 SVGs to guile_rplot_*.svg\n")
