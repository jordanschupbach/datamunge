(use-modules (datamunge))

(define iris (DataFrame-iris))
(format #t "iris: ~a rows x ~a cols\n\n" (DataFrame-nrows iris) (DataFrame-ncols iris))

(define model (new-RandomForestClassifier iris "Species ~ Petal.Length + Petal.Width"))
(RandomForestClassifier-print-summary model)

(format #t "\nConfusion matrix (rows = actual, cols = predicted):\n")
(format #t "~a\n" (DataFrame-to-string (RandomForestClassifier-confusion-matrix model)))

(format #t "\nMisclassified rows:\n")
(define predictions (RandomForestClassifier-predict model iris))
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

(Plot-save (RandomForestClassifier-plot-classification model iris "Petal.Length" "Petal.Width") "forest_iris_classification.svg")
(Plot-save (RandomForestClassifier-plot-decision-regions model "Petal.Length" "Petal.Width") "forest_iris_decision_regions.svg")
(format #t "\nSaved forest_iris_classification.svg and forest_iris_decision_regions.svg\n")

(define small-forest (new-RandomForestClassifier iris "Species ~ Petal.Length + Petal.Width" 5))
(format #t "\n5-tree forest:   training accuracy=~a%  OOB accuracy=~a%\n" (* 100.0 (RandomForestClassifier-training-accuracy small-forest)) (* 100.0 (RandomForestClassifier-oob-accuracy small-forest)))
(format #t "100-tree forest: training accuracy=~a%  OOB accuracy=~a%\n" (* 100.0 (RandomForestClassifier-training-accuracy model)) (* 100.0 (RandomForestClassifier-oob-accuracy model)))
(Plot-save (RandomForestClassifier-plot-decision-regions small-forest "Petal.Length" "Petal.Width") "forest_iris_decision_regions_5trees.svg")
(format #t "Saved forest_iris_decision_regions_5trees.svg\n")
