;; CMD: just TARGET=lda_iris_ex run-guile
;; Linear discriminant analysis on iris: summary, confusion matrix, predictions and discriminant plot.
(use-modules (oop goops) (datamunge) ((datamunge data) #:select (iris)))

(define iris-df (iris))
(format #t "iris: ~a rows x ~a cols\n\n" (nrows iris-df) (ncols iris-df))

(define model (make <lda> iris-df "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width"))
(print-summary model)

(format #t "\nConfusion matrix (rows = actual, cols = predicted):\n~a\n"
        (to-string (confusion-matrix model)))

(define newdata (make <data-frame>))
(add-numeric-column newdata "Sepal.Length" '(5.1 6.0 6.5 6.2))
(add-numeric-column newdata "Sepal.Width" '(3.5 2.7 3.0 2.8))
(add-numeric-column newdata "Petal.Length" '(1.4 4.5 5.5 4.8))
(add-numeric-column newdata "Petal.Width" '(0.2 1.5 2.0 1.8))
(format #t "\nPredictions for new flowers:\n~a\n" (to-string (predict-frame model newdata)))

(save-discriminant-plot model "lda_iris_discriminants.svg")
(display "\nSaved discriminant plot as lda_iris_discriminants.svg\n")
