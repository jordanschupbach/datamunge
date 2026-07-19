(use-modules (datamunge))

(define iris (DataFrame-iris))
(format #t "iris: ~a rows x ~a cols\n\n" (DataFrame-nrows iris) (DataFrame-ncols iris))

(define model (new-KNNClassifier iris "Species ~ Petal.Length + Petal.Width"))
(KNNClassifier-print-summary model)

(format #t "\nConfusion matrix (leave-one-out; rows = actual, cols = predicted):\n")
(format #t "~a\n" (DataFrame-to-string (KNNClassifier-confusion-matrix model)))

(format #t "\nLeave-one-out misclassified rows:\n")
(define fitted (KNNClassifier-predict model iris))
(define misclassified 0)
(define n (DataFrame-nrows iris))
(do ((i 0 (+ i 1))) ((= i n))
  (let* ((actual (DataFrame-string-at iris "Species" i))
         (pred (vector-ref fitted i)))
    (unless (string=? pred actual)
      (set! misclassified (+ misclassified 1))
      (format #t "  row ~a: Petal.Length=~a Petal.Width=~a  actual=~a  predicted=~a\n" i
              (DataFrame-numeric-at iris "Petal.Length" i) (DataFrame-numeric-at iris "Petal.Width" i) actual pred))))
(format #t "~a of ~a misclassified (~a%)\n" misclassified n (* 100.0 (/ misclassified n)))

(Plot-save (KNNClassifier-plot-decision-regions model "Petal.Length" "Petal.Width") "knn_iris_decision_regions_k5.svg")
(format #t "\nSaved knn_iris_decision_regions_k5.svg\n")

(define k1 (new-KNNClassifier iris "Species ~ Petal.Length + Petal.Width" 1))
(format #t "\nk=1  leave-one-out accuracy: ~a%\n" (* 100.0 (KNNClassifier-training-accuracy k1)))
(Plot-save (KNNClassifier-plot-decision-regions k1 "Petal.Length" "Petal.Width") "knn_iris_decision_regions_k1.svg")
(format #t "Saved knn_iris_decision_regions_k1.svg\n")

(define k25 (new-KNNClassifier iris "Species ~ Petal.Length + Petal.Width" 25))
(format #t "\nk=25 leave-one-out accuracy: ~a%\n" (* 100.0 (KNNClassifier-training-accuracy k25)))
(Plot-save (KNNClassifier-plot-decision-regions k25 "Petal.Length" "Petal.Width") "knn_iris_decision_regions_k25.svg")
(format #t "Saved knn_iris_decision_regions_k25.svg\n")
