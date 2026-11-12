;;;; R base-graphics-style plotting with dm:r-plot / dm:r-layout: plot, hist, barplot, boxplot, pie, qqnorm, layouts.
(require "asdf")
(asdf:load-system "datamunge")

;; plot(x, y, type = "p") then abline() layered on afterward.
(defparameter *scatter*
  (dm:r-plot-plot #(1d0 2d0 3d0 4d0 5d0) #(2.1d0 3.9d0 6.2d0 7.8d0 10.1d0) "p" "observed"))
(dm:abline *scatter* 0d0 2d0 (dm:make-rgb 220 38 38) 1.5d0)
(dm:y-label (dm:x-label (dm:title *scatter* "plot() + abline()") "x") "y")
(dm:save-svg *scatter* "lisp_rplot_scatter.svg")

;; hist(): equal-width binning over the data range.
(defparameter *hist*
  (dm:r-plot-hist #(1d0 2d0 2d0 3d0 3d0 3d0 4d0 4d0 4d0 4d0 5d0 5d0 5d0 6d0 6d0 7d0) 6 "counts"))
(dm:save-svg *hist* "lisp_rplot_hist.svg")

;; barplot(): categorical positions with x tick labels.
(let ((bars (dm:r-plot-barplot #(23d0 41d0 12d0) #("A" "B" "C"))))
  (dm:title bars "barplot()")
  (dm:save-svg bars "lisp_rplot_barplot.svg"))

;; boxplot(): Tukey five-number summary per group.
(let ((box (dm:r-plot-boxplot (list #(2d0 4d0 4d0 4d0 5d0 5d0 7d0 9d0)
                                    #(1d0 2d0 2d0 2d0 3d0 3d0 3d0 3d0 4d0 20d0))
                              #("low variance" "has outlier"))))
  (dm:title box "boxplot()")
  (dm:save-svg box "lisp_rplot_boxplot.svg"))

;; pie(): wedge areas proportional to value, axes hidden automatically.
(let ((pie (dm:r-plot-pie #(35d0 25d0 20d0 20d0) #("Q1" "Q2" "Q3" "Q4"))))
  (dm:title pie "pie()")
  (dm:save-svg pie "lisp_rplot_pie.svg"))

;; curve() is omitted to stay in step with the Lua example (Lua cannot supply the callback).

;; qqnorm() + qqline(): standard-normal Q-Q plot with a fitted reference line.
(let* ((residuals #(-2.1d0 -1.3d0 -0.8d0 -0.4d0 -0.1d0 0.2d0 0.5d0 0.9d0 1.4d0 2.3d0))
       (qq (dm:r-plot-qqnorm residuals)))
  (dm:qqline qq residuals)
  (dm:save-svg qq "lisp_rplot_qqnorm.svg"))

;; par(mfrow = c(1, 2))-style multi-panel composition via r-layout.
(let ((layout (dm:r-layout-create 1 2)))
  (dm:add layout *scatter*)
  (dm:add layout *hist*)
  (dm:save-svg layout "lisp_rplot_layout.svg"))

(format t "Wrote 7 SVGs to lisp_rplot_*.svg~%")
