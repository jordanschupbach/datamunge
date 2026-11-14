;; CMD: just TARGET=ggplot_ex run-guile
;; Demonstrates GGPlot -- the ggplot2-style grammar-of-graphics library.
(use-modules (oop goops) (datamunge) ((datamunge data) #:select (iris)))

(define iris-df (iris))

;; Every GGPlot method returns the plot itself, so (chain p f ...) mimics C++/Ruby method
;; chaining: each step is a procedure applied to the plot in turn.
(define (chain p . steps)
  (for-each (lambda (step) (step p)) steps)
  p)

;; ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Species)) + geom_point()
(chain (make <gg-plot> iris-df "Sepal.Length" "Sepal.Width" "Species")
       geom-point
       (lambda (p) (labs p "Iris Sepal Dimensions" "Sepal Length" "Sepal Width"))
       theme-minimal
       (lambda (p) (save-svg p "guile_ggplot_point.svg")))

;; + geom_smooth(): an lm() fit line, reusing stats::LM internally.
(chain (make <gg-plot> iris-df "Sepal.Length" "Petal.Length")
       (lambda (p) (geom-point p (make <rgb> 156 163 175) 2.5))
       geom-smooth
       (lambda (p) (labs p "Petal Length vs Sepal Length With a Linear Fit" "Sepal Length" "Petal Length"))
       (lambda (p) (save-svg p "guile_ggplot_smooth.svg")))

;; geom_bar(): counts a discrete column (stat = "count").
(chain (make <gg-plot> iris-df "Species")
       geom-bar
       (lambda (p) (labs p "Observations per Species" "Species" "Count"))
       theme-bw
       (lambda (p) (save-svg p "guile_ggplot_bar.svg")))

;; geom_boxplot(): grouped by a discrete x column.
(chain (make <gg-plot> iris-df "Species" "Petal.Width")
       geom-boxplot
       (lambda (p) (labs p "Petal Width by Species" "Species" "Petal Width"))
       (lambda (p) (save-svg p "guile_ggplot_boxplot.svg")))

;; geom_histogram() + geom_density(): distribution of a single numeric column.
(chain (make <gg-plot> iris-df "Sepal.Length")
       (lambda (p) (geom-histogram p 20))
       (lambda (p) (labs p "Distribution of Sepal Length" "Sepal Length" "Count"))
       (lambda (p) (save-svg p "guile_ggplot_histogram.svg")))

(chain (make <gg-plot> iris-df "Sepal.Length")
       geom-density
       (lambda (p) (labs p "Density of Sepal Length" "Sepal Length" "Density"))
       (lambda (p) (save-svg p "guile_ggplot_density.svg")))

;; facet_wrap(): one panel per Species, composed via RLayout under the hood.
(chain (make <gg-plot> iris-df "Petal.Length" "Petal.Width")
       geom-point
       (lambda (p) (facet-wrap p "Species"))
       (lambda (p) (labs p "Petal Dimensions" "Petal Length" "Petal Width"))
       (lambda (p) (save-svg p "guile_ggplot_facet.svg")))

;; theme_classic(): another built-in theme.
(chain (make <gg-plot> iris-df "Sepal.Length" "Sepal.Width" "Species")
       geom-point
       theme-classic
       (lambda (p) (save-svg p "guile_ggplot_classic.svg")))

(display "Wrote 7 SVGs to guile_ggplot_*.svg\n")
