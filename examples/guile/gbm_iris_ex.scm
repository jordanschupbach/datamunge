;; CMD: just TARGET=gbm_iris_ex run-guile
;; Gradient-boosted trees (GBM) on iris: summary, confusion matrix, misclassified rows, 5 vs 100 rounds.
(use-modules (oop goops) (srfi srfi-1) (datamunge) ((datamunge data) #:select (iris)))

(define iris-data (iris))
(format #t "iris: ~a rows x ~a cols\n\n" (nrows iris-data) (ncols iris-data))

(define model (make <gbm-classifier> iris-data "Species ~ Petal.Length + Petal.Width"))
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

(save (plot-training-deviance model) "gbm_iris_training_deviance.svg")
(save (plot-decision-regions model "Petal.Length" "Petal.Width") "gbm_iris_decision_regions.svg")
(display "\nSaved gbm_iris_training_deviance.svg and gbm_iris_decision_regions.svg\n")

;; A handful of boosting rounds vs a well-boosted ensemble: each additional round chips away
;; at the training loss, gradually sharpening the decision boundary.
;; (make <gbm-classifier> data formula n-trees learning-rate max-depth min-samples-split
;;                        min-samples-leaf subsample seed)
(define few (make <gbm-classifier> iris-data "Species ~ Petal.Length + Petal.Width" 5))
(format #t "\n5-round ensemble:   training accuracy=~a%  deviance=~a\n"
        (* 100 (training-accuracy few)) (last (training-deviance few)))
(format #t "100-round ensemble: training accuracy=~a%  deviance=~a\n"
        (* 100 (training-accuracy model)) (last (training-deviance model)))
(save (plot-decision-regions few "Petal.Length" "Petal.Width") "gbm_iris_decision_regions_5rounds.svg")
(display "Saved gbm_iris_decision_regions_5rounds.svg\n")
