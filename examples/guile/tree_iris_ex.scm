;; CMD: just TARGET=tree_iris_ex run-guile
;; Decision tree classifier on iris: summary, confusion matrix, misclassified rows and plots.
(use-modules (oop goops) (datamunge) ((datamunge data) #:select (iris)))

(define iris-data (iris))
(format #t "iris: ~a rows x ~a cols\n\n" (nrows iris-data) (ncols iris-data))

(define model (make <decision-tree-classifier> iris-data "Species ~ Petal.Length + Petal.Width"))
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

(save (plot-classification model iris-data "Petal.Length" "Petal.Width") "tree_iris_classification.svg")
(save (plot-decision-regions model "Petal.Length" "Petal.Width") "tree_iris_decision_regions.svg")
(display "\nSaved tree_iris_classification.svg and tree_iris_decision_regions.svg\n")

;; A shallower tree, for comparison, showing a coarser (but still fairly accurate) decision boundary.
;; (make <decision-tree-classifier> data formula max-depth min-samples-split min-samples-leaf criterion)
(define shallow (make <decision-tree-classifier> iris-data "Species ~ Petal.Length + Petal.Width" 2))
(format #t "\nDepth-2 tree training accuracy: ~a% (~a leaves)\n"
        (* (training-accuracy shallow) 100) (leaf-count shallow))
(save (plot-decision-regions shallow "Petal.Length" "Petal.Width") "tree_iris_decision_regions_depth2.svg")
(display "Saved tree_iris_decision_regions_depth2.svg\n")
