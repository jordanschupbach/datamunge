;; Demonstrates RPlot / RLayout -- the R-base-graphics-style plotting library.
(use-modules (datamunge))

(define (rgb r g b)
  (let ((c (new-RGB)))
    (RGB-r-set c r)
    (RGB-g-set c g)
    (RGB-b-set c b)
    c))

(define (dvv rows)
  (let ((m (new-DVectorVector)))
    (for-each (lambda (r) (DVectorVector-push! m r)) rows)
    m))

;; plot(x, y, type = "p") then abline() layered on afterward.
(define scatter (RPlot-plot (list 1.0 2.0 3.0 4.0 5.0) (list 2.1 3.9 6.2 7.8 10.1) "p" "observed"))
(RPlot-abline scatter 0.0 2.0 (rgb 220 38 38) 1.5)
(Plot-title scatter "plot() + abline()")
(Plot-x-label scatter "x")
(Plot-y-label scatter "y")
(Plot-save-svg scatter "guile_rplot_scatter.svg")

;; hist(): equal-width binning over the data range.
(define hist (RPlot-hist (list 1 2 2 3 3 3 4 4 4 4 5 5 5 6 6 7) 6 "counts"))
(Plot-save-svg hist "guile_rplot_hist.svg")

;; barplot(): categorical positions with x tick labels.
(define bars (RPlot-barplot (list 23.0 41.0 12.0) (list "A" "B" "C")))
(Plot-title bars "barplot()")
(Plot-save-svg bars "guile_rplot_barplot.svg")

;; boxplot(): Tukey five-number summary per group.
(define groups (dvv (list (list 2 4 4 4 5 5 7 9) (list 1 2 2 2 3 3 3 3 4 20))))
(define box (RPlot-boxplot groups (list "low variance" "has outlier")))
(Plot-title box "boxplot()")
(Plot-save-svg box "guile_rplot_boxplot.svg")

;; pie(): wedge areas proportional to value, axes hidden automatically.
(define pie (RPlot-pie (list 35.0 25.0 20.0 20.0) (list "Q1" "Q2" "Q3" "Q4")))
(Plot-title pie "pie()")
(Plot-save-svg pie "guile_rplot_pie.svg")

;; curve() is omitted here: it samples a Callback (a SWIG director), which the Guile binding
;; cannot subclass. Every other RPlot method below is director-free.

;; qqnorm() + qqline(): standard-normal Q-Q plot with a fitted reference line.
(define residuals (list -2.1 -1.3 -0.8 -0.4 -0.1 0.2 0.5 0.9 1.4 2.3))
(define qq (RPlot-qqnorm residuals))
(RPlot-qqline qq residuals)
(Plot-save-svg qq "guile_rplot_qqnorm.svg")

;; par(mfrow = c(1, 2))-style multi-panel composition via RLayout.
(define layout (RLayout-create 1 2))
(RLayout-add layout scatter)
(RLayout-add layout hist)
(RLayout-save-svg layout "guile_rplot_layout.svg")

(format #t "Wrote 7 SVGs to guile_rplot_*.svg\n")
