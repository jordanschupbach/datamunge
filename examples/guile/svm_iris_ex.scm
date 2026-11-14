;; CMD: just TARGET=svm_iris_ex run-guile
;; Support vector machine on iris: RBF vs. linear kernels, confusion matrix and predictions.
(use-modules (oop goops) (datamunge) ((datamunge data) #:select (iris)))

(define formula "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width")
(define iris-df (iris))
(format #t "iris: ~a rows x ~a cols\n\n" (nrows iris-df) (ncols iris-df))

(display "=== RBF kernel (default) ===\n")
(define rbf-model (make <svm> iris-df formula))
(print-summary rbf-model)
(format #t "\nConfusion matrix (rows = actual, cols = predicted):\n~a\n"
        (to-string (confusion-matrix rbf-model)))

(display "\n=== Linear kernel, for comparison ===\n")
(define linear-model (make <svm> iris-df formula "linear"))
(format #t "Training accuracy: ~a%\n" (* (training-accuracy linear-model) 100))
(format #t "Support vectors: ~a\n" (num-support-vectors linear-model))

(define newdata (make <data-frame>))
(add-numeric-column newdata "Sepal.Length" '(5.1 6.0 6.5 6.2))
(add-numeric-column newdata "Sepal.Width" '(3.5 2.7 3.0 2.8))
(add-numeric-column newdata "Petal.Length" '(1.4 4.5 5.5 4.8))
(add-numeric-column newdata "Petal.Width" '(0.2 1.5 2.0 1.8))
(format #t "\nRBF predictions for new flowers (votes out of 3 one-vs-one pairs):\n~a\n"
        (to-string (predict-frame rbf-model newdata)))
