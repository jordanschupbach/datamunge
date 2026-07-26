;; Demonstrates GGPlot -- the ggplot2-style grammar-of-graphics library.
(use-modules (datamunge))

(define (rgb r g b)
  (let ((c (new-RGB)))
    (RGB-r-set c r)
    (RGB-g-set c g)
    (RGB-b-set c b)
    c))

(define iris (DataFrame-iris))

;; ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Species)) + geom_point()
(define scatter (new-GGPlot iris "Sepal.Length" "Sepal.Width" "Species"))
(GGPlot-geom-point scatter)
(GGPlot-labs scatter "Iris Sepal Dimensions" "Sepal Length" "Sepal Width")
(GGPlot-theme-minimal scatter)
(GGPlot-save-svg scatter "guile_ggplot_point.svg")

;; + geom_smooth(): an lm() fit line, reusing stats::LM internally.
(define smooth (new-GGPlot iris "Sepal.Length" "Petal.Length"))
(GGPlot-geom-point smooth (rgb 156 163 175) 2.5)
(GGPlot-geom-smooth smooth)
(GGPlot-labs smooth "Petal Length vs Sepal Length With a Linear Fit" "Sepal Length" "Petal Length")
(GGPlot-save-svg smooth "guile_ggplot_smooth.svg")

;; geom_bar(): counts a discrete column (stat = "count").
(define bar (new-GGPlot iris "Species"))
(GGPlot-geom-bar bar)
(GGPlot-labs bar "Observations per Species" "Species" "Count")
(GGPlot-theme-bw bar)
(GGPlot-save-svg bar "guile_ggplot_bar.svg")

;; geom_boxplot(): grouped by a discrete x column.
(define box (new-GGPlot iris "Species" "Petal.Width"))
(GGPlot-geom-boxplot box)
(GGPlot-labs box "Petal Width by Species" "Species" "Petal Width")
(GGPlot-save-svg box "guile_ggplot_boxplot.svg")

;; geom_histogram() + geom_density(): distribution of a single numeric column.
(define hist (new-GGPlot iris "Sepal.Length"))
(GGPlot-geom-histogram hist 20)
(GGPlot-labs hist "Distribution of Sepal Length" "Sepal Length" "Count")
(GGPlot-save-svg hist "guile_ggplot_histogram.svg")

(define density (new-GGPlot iris "Sepal.Length"))
(GGPlot-geom-density density)
(GGPlot-labs density "Density of Sepal Length" "Sepal Length" "Density")
(GGPlot-save-svg density "guile_ggplot_density.svg")

;; facet_wrap(): one panel per Species, composed via RLayout under the hood.
(define faceted (new-GGPlot iris "Petal.Length" "Petal.Width"))
(GGPlot-geom-point faceted)
(GGPlot-facet-wrap faceted "Species")
(GGPlot-labs faceted "Petal Dimensions" "Petal Length" "Petal Width")
(GGPlot-save-svg faceted "guile_ggplot_facet.svg")

;; theme_classic(): another built-in theme. (scale_color_manual is omitted here -- the
;; std::vector<RGB> overload it needs is not part of the Guile binding's exposed surface.)
(define classic (new-GGPlot iris "Sepal.Length" "Sepal.Width" "Species"))
(GGPlot-geom-point classic)
(GGPlot-theme-classic classic)
(GGPlot-save-svg classic "guile_ggplot_classic.svg")

(format #t "Wrote 7 SVGs to guile_ggplot_*.svg\n")
