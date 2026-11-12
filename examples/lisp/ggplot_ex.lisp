;;;; Demonstrates dm:gg-plot -- the ggplot2-style grammar-of-graphics library.
(require "asdf")
(asdf:load-system "datamunge")

(defun rgb (r g b)
  (let ((c (dm:make-rgb)))
    (setf (dm:r c) r (dm:g c) g (dm:b c) b)
    c))

(defparameter *iris* (dm:data-frame-iris))

;; ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Species)) + geom_point()
(let ((scatter (dm:make-gg-plot *iris* "Sepal.Length" "Sepal.Width" "Species")))
  (dm:geom-point scatter)
  (dm:labs scatter "Iris Sepal Dimensions" "Sepal Length" "Sepal Width")
  (dm:theme-minimal scatter)
  (dm:save-svg scatter "lisp_ggplot_point.svg"))

;; + geom_smooth(): an lm() fit line, reusing stats::LM internally.
(let ((smooth (dm:make-gg-plot *iris* "Sepal.Length" "Petal.Length")))
  (dm:geom-point smooth (rgb 156 163 175) 2.5d0)
  (dm:geom-smooth smooth)
  (dm:labs smooth "Petal Length vs Sepal Length With a Linear Fit" "Sepal Length" "Petal Length")
  (dm:save-svg smooth "lisp_ggplot_smooth.svg"))

;; geom_bar(): counts a discrete column (stat = "count").
(let ((bar (dm:make-gg-plot *iris* "Species")))
  (dm:geom-bar bar)
  (dm:labs bar "Observations per Species" "Species" "Count")
  (dm:theme-bw bar)
  (dm:save-svg bar "lisp_ggplot_bar.svg"))

;; geom_boxplot(): grouped by a discrete x column.
(let ((box (dm:make-gg-plot *iris* "Species" "Petal.Width")))
  (dm:geom-boxplot box)
  (dm:labs box "Petal Width by Species" "Species" "Petal Width")
  (dm:save-svg box "lisp_ggplot_boxplot.svg"))

;; geom_histogram() + geom_density(): distribution of a single numeric column.
(let ((hist (dm:make-gg-plot *iris* "Sepal.Length")))
  (dm:geom-histogram hist 20)
  (dm:labs hist "Distribution of Sepal Length" "Sepal Length" "Count")
  (dm:save-svg hist "lisp_ggplot_histogram.svg"))

(let ((density (dm:make-gg-plot *iris* "Sepal.Length")))
  (dm:geom-density density)
  (dm:labs density "Density of Sepal Length" "Sepal Length" "Density")
  (dm:save-svg density "lisp_ggplot_density.svg"))

;; facet_wrap(): one panel per Species, composed via RLayout under the hood.
(let ((faceted (dm:make-gg-plot *iris* "Petal.Length" "Petal.Width")))
  (dm:geom-point faceted)
  (dm:facet-wrap faceted "Species")
  (dm:labs faceted "Petal Dimensions" "Petal Length" "Petal Width")
  (dm:save-svg faceted "lisp_ggplot_facet.svg"))

;; theme_classic(): another built-in theme.
(let ((classic (dm:make-gg-plot *iris* "Sepal.Length" "Sepal.Width" "Species")))
  (dm:geom-point classic)
  (dm:theme-classic classic)
  (dm:save-svg classic "lisp_ggplot_classic.svg"))

(format t "Wrote 7 SVGs to lisp_ggplot_*.svg~%")
