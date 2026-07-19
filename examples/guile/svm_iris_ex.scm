(use-modules (datamunge))

(define iris (DataFrame-iris))
(format #t "iris: ~a rows x ~a cols\n\n" (DataFrame-nrows iris) (DataFrame-ncols iris))

(format #t "=== RBF kernel (default) ===\n")
(define rbf-model (new-SVM iris "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width"))
(SVM-print-summary rbf-model)
(format #t "\nConfusion matrix (rows = actual, cols = predicted):\n")
(format #t "~a\n" (DataFrame-to-string (SVM-confusion-matrix rbf-model)))

(format #t "\n=== Linear kernel, for comparison ===\n")
(define linear-model (new-SVM iris "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width" "linear"))
(format #t "Training accuracy: ~a%\n" (* 100.0 (SVM-training-accuracy linear-model)))
(format #t "Support vectors: ~a\n" (SVM-num-support-vectors linear-model))

(define newdata (DataFrame-empty))
(DataFrame-add-numeric-column newdata "Sepal.Length" (list 5.1 6.0 6.5 6.2))
(DataFrame-add-numeric-column newdata "Sepal.Width" (list 3.5 2.7 3.0 2.8))
(DataFrame-add-numeric-column newdata "Petal.Length" (list 1.4 4.5 5.5 4.8))
(DataFrame-add-numeric-column newdata "Petal.Width" (list 0.2 1.5 2.0 1.8))

(format #t "\nRBF predictions for new flowers (votes out of 3 one-vs-one pairs):\n")
(format #t "~a\n" (DataFrame-to-string (SVM-predict-frame rbf-model newdata)))
