;; CMD: just TARGET=xgboost_iris_ex run-guile
;; XGBoost classifier on iris: summary, confusion matrix, misclassified rows, L2-regularization comparison.
(use-modules (oop goops) (srfi srfi-1) (datamunge) ((datamunge data) #:select (iris)))

(define iris-data (iris))
(format #t "iris: ~a rows x ~a cols\n\n" (nrows iris-data) (ncols iris-data))

(define model (make <xg-boost-classifier> iris-data "Species ~ Petal.Length + Petal.Width"))
(print-summary model)

(format #t "\nConfusion matrix (rows = actual, cols = predicted):\n~a\n" (to-string (confusion-matrix model)))

(display "\nMisclassified rows:\n")
(define predictions (predict model iris-data))
(define n (nrows iris-data))
(define misclassified 0)
(let loop ((i 0) (preds predictions))
  (when (< i n)
    (let ((actual (string-at iris-data "Species" i))
          (predicted (car preds)))
      (unless (string=? predicted actual)
        (set! misclassified (+ misclassified 1))
        (format #t "  row ~a: Petal.Length=~a Petal.Width=~a  actual=~a  predicted=~a\n"
                i (numeric-at iris-data "Petal.Length" i) (numeric-at iris-data "Petal.Width" i)
                actual predicted)))
    (loop (+ i 1) (cdr preds))))
(format #t "~a of ~a misclassified (~a%)\n" misclassified n (/ (* 100.0 misclassified) n))

(save (plot-training-deviance model) "xgboost_iris_training_deviance.svg")
(save (plot-decision-regions model "Petal.Length" "Petal.Width") "xgboost_iris_decision_regions.svg")
(display "\nSaved xgboost_iris_training_deviance.svg and xgboost_iris_decision_regions.svg\n")

;; XGBoost's defining lever is regularization: a heavily L2-regularized model (large lambda)
;; grows the same deep trees but keeps every leaf weight small, producing a much smoother
;; decision boundary than the lightly-regularized default.
;; (make <xg-boost-classifier> data formula n-trees learning-rate max-depth lambda alpha gamma
;;                             min-child-weight min-samples-leaf subsample colsample-bytree seed)
(define heavy (make <xg-boost-classifier> iris-data "Species ~ Petal.Length + Petal.Width" 100 0.3 6 50.0))
(format #t "\nlambda=1 (default):   training accuracy=~a%  deviance=~a\n"
        (* 100 (training-accuracy model)) (last (training-deviance model)))
(format #t "lambda=50 (heavy L2): training accuracy=~a%  deviance=~a\n"
        (* 100 (training-accuracy heavy)) (last (training-deviance heavy)))
(save (plot-decision-regions heavy "Petal.Length" "Petal.Width") "xgboost_iris_decision_regions_heavy_lambda.svg")
(display "Saved xgboost_iris_decision_regions_heavy_lambda.svg\n")
