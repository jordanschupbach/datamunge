;; CMD: just TARGET=forest_iris_ex run-guile
;; Random forest classifier on iris: summary, confusion matrix, misclassifications, OOB accuracy.
(use-modules (oop goops) (datamunge) ((datamunge data) #:select (iris)))

(define iris-data (iris))
(format #t "iris: ~a rows x ~a cols\n\n" (nrows iris-data) (ncols iris-data))

(define model (make <random-forest-classifier> iris-data "Species ~ Petal.Length + Petal.Width"))
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

(save (plot-classification model iris-data "Petal.Length" "Petal.Width") "forest_iris_classification.svg")
(save (plot-decision-regions model "Petal.Length" "Petal.Width") "forest_iris_decision_regions.svg")
(display "\nSaved forest_iris_classification.svg and forest_iris_decision_regions.svg\n")

;; A small forest, for comparison, showing how out-of-bag accuracy stabilizes as more trees
;; are added -- the defining random forest effect a single decision tree cannot demonstrate.
;; (make <random-forest-classifier> data formula n-trees max-depth min-samples-split min-samples-leaf
;;                                  max-features criterion bootstrap sample-fraction seed)
(define small-forest (make <random-forest-classifier> iris-data "Species ~ Petal.Length + Petal.Width" 5))
(format #t "\n5-tree forest:   training accuracy=~a%  OOB accuracy=~a%\n"
        (* (training-accuracy small-forest) 100) (* (oob-accuracy small-forest) 100))
(format #t "100-tree forest: training accuracy=~a%  OOB accuracy=~a%\n"
        (* (training-accuracy model) 100) (* (oob-accuracy model) 100))
(save (plot-decision-regions small-forest "Petal.Length" "Petal.Width") "forest_iris_decision_regions_5trees.svg")
(display "Saved forest_iris_decision_regions_5trees.svg\n")
