;; CMD: just TARGET=glm_iris_ex run-guile
;; Generalized linear models on iris: logistic regression (binomial) and Poisson regression.
(use-modules (oop goops) (datamunge) ((datamunge data) #:select (iris)))

(define iris-data (iris))
(define rows (iota (nrows iris-data)))
(define (column name idx) (map (lambda (i) (numeric-at iris-data name i)) idx))

;; Logistic regression: versicolor vs virginica only -- setosa is perfectly separable from
;; the other two on these predictors, which sends logistic regression's coefficients toward
;; +/-infinity (a genuine degeneracy of the method, not a bug) -- so this is the well-behaved
;; binary split for a demo.
(define kept
  (filter (lambda (i) (member (string-at iris-data "Species" i) '("versicolor" "virginica"))) rows))
(define is-virginica
  (map (lambda (i) (if (string=? (string-at iris-data "Species" i) "virginica") 1.0 0.0)) kept))
(define petal-length (column "Petal.Length" kept))
(define petal-width (column "Petal.Width" kept))

(define sub (make <data-frame>))
(add-numeric-column sub "Petal.Length" petal-length)
(add-numeric-column sub "Petal.Width" petal-width)
(add-numeric-column sub "is_virginica" is-virginica)

(display "=================== Logistic regression (binomial, logit link) ===================\n")
(define logit (make <glm> sub "is_virginica ~ Petal.Length + Petal.Width" "binomial"))
(print-summary logit)

(define n (length is-virginica))
(define correct
  (length (filter identity
                  (map (lambda (f y) (eq? (>= f 0.5) (>= y 0.5))) (fitted-values logit) is-virginica))))
(format #t "\nResubstitution accuracy at 0.5 threshold: ~a%\n" (/ (* 100.0 correct) n))

(save-diagnostic-plots logit "glm_logistic_iris")
(display "\nSaved glm_logistic_iris_{residuals_vs_fitted,normal_qq,scale_location,residuals_vs_leverage}.svg\n")

;; Predicted-probability curve across Petal.Length, with Petal.Width held at its mean -- the
;; classic sigmoid shape of a fitted logistic regression, with a 95% confidence band.
(define grid-n 100)
(define width-mean (/ (apply + petal-width) (length petal-width)))
(define pl-min (- (apply min petal-length) 0.3))
(define pl-max (+ (apply max petal-length) 0.3))
(define grid (make <data-frame>))
(add-numeric-column grid "Petal.Length"
                    (map (lambda (i) (+ pl-min (/ (* (- pl-max pl-min) i) (- grid-n 1)))) (iota grid-n)))
(add-numeric-column grid "Petal.Width" (make-list grid-n width-mean))
(define curve-frame (predict-frame logit grid "confidence"))
(display "\nPredicted-probability curve (first 5 rows):\n")
(format #t "~a\n" (to-string curve-frame 5))

;; Poisson regression, for contrast: same IRLS engine, different family/link.
(display "\n=================== Poisson regression (log link) ===================\n")
(define count-data (make <data-frame>))
(add-numeric-column count-data "Sepal.Width" (column "Sepal.Width" rows))
(add-numeric-column count-data "Petal.Length" (column "Petal.Length" rows))
(add-numeric-column count-data "count"
                    (map (lambda (x) (floor (+ x 0.5))) (column "Sepal.Length" rows)))
(print-summary (make <glm> count-data "count ~ Sepal.Width + Petal.Length" "poisson"))
