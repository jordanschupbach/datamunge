;; CMD: just TARGET=bspline_ex run-guile
;; Tensor-product B-spline regression: fit z ~ bs(x, y) to a smooth surface and predict.
(use-modules (oop goops) (datamunge) (srfi srfi-1) (ice-9 format))

;; Sample a smooth surface z = sin(x) + y^2 on an 8x8 grid over [0, 1]^2.
(define grid
  (append-map (lambda (iy) (map (lambda (ix) (cons (/ ix 7.0) (/ iy 7.0))) (iota 8)))
              (iota 8)))
(define x (map car grid))
(define y (map cdr grid))
(define z (map (lambda (xv yv) (+ (sin xv) (* yv yv))) x y))

(define data (make <data-frame>))
(add-numeric-column data "x" x)
(add-numeric-column data "y" y)
(add-numeric-column data "z" z)

(define surface (make <lm> data "z ~ bs(x, y)"))
(display "Fitted z ~ bs(x, y)\n")
(print-summary surface)

(define new-points (make <data-frame>))
(add-numeric-column new-points "x" '(0.25 0.75))
(add-numeric-column new-points "y" '(0.50 0.25))
(define pred (predict surface new-points))
(format #t "Predictions: [~,6f, ~,6f]\n" (car pred) (cadr pred))
