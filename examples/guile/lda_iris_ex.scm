(use-modules (datamunge))

(define iris (DataFrame-iris))
(format #t "iris: ~a rows x ~a cols\n\n" (DataFrame-nrows iris) (DataFrame-ncols iris))

(define model (new-LDA iris "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width"))
(LDA-print-summary model)

(format #t "\nConfusion matrix (rows = actual, cols = predicted):\n")
(format #t "~a\n" (DataFrame-to-string (LDA-confusion-matrix model)))

(define newdata (DataFrame-empty))
(DataFrame-add-numeric-column newdata "Sepal.Length" (list 5.1 6.0 6.5 6.2))
(DataFrame-add-numeric-column newdata "Sepal.Width" (list 3.5 2.7 3.0 2.8))
(DataFrame-add-numeric-column newdata "Petal.Length" (list 1.4 4.5 5.5 4.8))
(DataFrame-add-numeric-column newdata "Petal.Width" (list 0.2 1.5 2.0 1.8))

(format #t "\nPredictions for new flowers:\n")
(format #t "~a\n" (DataFrame-to-string (LDA-predict-frame model newdata)))

(LDA-save-discriminant-plot model "lda_iris_discriminants.svg")
(format #t "\nSaved discriminant plot as lda_iris_discriminants.svg\n")
