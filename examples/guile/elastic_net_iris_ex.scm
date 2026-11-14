;; CMD: just TARGET=elastic_net_iris_ex run-guile
;; Ridge, Lasso and Elastic Net on iris, with coefficient-path and CV-curve figures.
(use-modules (oop goops) (datamunge) ((datamunge data) #:select (iris)))

(define formula "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width + Sepal.Length:Petal.Width")

(define iris-df (iris))
(format #t "iris: ~a rows x ~a cols\n" (nrows iris-df) (ncols iris-df))
(format #t "formula: ~a\n\n" formula)

(display "=================== Ridge ===================\n")
(define ridge (make <ridge> iris-df formula))
(print-summary ridge)

(display "\n=================== Lasso ===================\n")
(define lasso (make <lasso> iris-df formula))
(print-summary lasso)

(display "\n================= Elastic Net =================\n")
;; (make <elastic-net> data formula #:optional alpha lambda n-lambda cv-folds standardize seed)
(define elastic (make <elastic-net> iris-df formula 0.5))
(print-summary elastic)

(display "\nSaved figures showing how each model's coefficients respond to the regularization strength, and the cross-validation curve used to pick it:\n")

(for-each
 (lambda (entry)
   (let ((name (car entry)) (model (cadr entry)) (label (caddr entry)))
     (save (plot-coefficient-path model) (format #f "elastic_net_~a_path.svg" name))
     (save (plot-cv-curve model) (format #f "elastic_net_~a_cv.svg" name))
     (format #t "  ~a elastic_net_~a_path.svg, elastic_net_~a_cv.svg\n" label name name)))
 (list (list "ridge" ridge "ridge:      ")
       (list "lasso" lasso "lasso:      ")
       (list "elasticnet" elastic "elastic net:")))
