(use-modules (datamunge))

(define iris (DataFrame-iris))
(format #t "iris: ~a rows x ~a cols\n\n" (DataFrame-nrows iris) (DataFrame-ncols iris))

(define model (new-DecisionTreeClassifier iris "Species ~ Petal.Length + Petal.Width"))
(DecisionTreeClassifier-print-summary model)

(format #t "\nConfusion matrix (rows = actual, cols = predicted):\n")
(format #t "~a\n" (DataFrame-to-string (DecisionTreeClassifier-confusion-matrix model)))

(format #t "\nMisclassified rows:\n")
(define predictions (DecisionTreeClassifier-predict model iris))
(define misclassified 0)
(define n (DataFrame-nrows iris))
(do ((i 0 (+ i 1))) ((= i n))
  (let* ((actual (DataFrame-string-at iris "Species" i))
         (pred (vector-ref predictions i)))
    (unless (string=? pred actual)
      (set! misclassified (+ misclassified 1))
      (format #t "  row ~a: Petal.Length=~a Petal.Width=~a  actual=~a  predicted=~a\n" i
              (DataFrame-numeric-at iris "Petal.Length" i) (DataFrame-numeric-at iris "Petal.Width" i) actual pred))))
(format #t "~a of ~a misclassified (~a%)\n" misclassified n (* 100.0 (/ misclassified n)))

(Plot-save (DecisionTreeClassifier-plot-classification model iris "Petal.Length" "Petal.Width") "tree_iris_classification.svg")
(Plot-save (DecisionTreeClassifier-plot-decision-regions model "Petal.Length" "Petal.Width") "tree_iris_decision_regions.svg")
(format #t "\nSaved tree_iris_classification.svg and tree_iris_decision_regions.svg\n")

(define shallow (new-DecisionTreeClassifier iris "Species ~ Petal.Length + Petal.Width" 2))
(format #t "\nDepth-2 tree training accuracy: ~a% (~a leaves)\n" (* 100.0 (DecisionTreeClassifier-training-accuracy shallow)) (DecisionTreeClassifier-leaf-count shallow))
(Plot-save (DecisionTreeClassifier-plot-decision-regions shallow "Petal.Length" "Petal.Width") "tree_iris_decision_regions_depth2.svg")
(format #t "Saved tree_iris_decision_regions_depth2.svg\n")
