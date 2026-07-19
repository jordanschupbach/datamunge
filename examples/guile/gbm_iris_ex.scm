(use-modules (datamunge))

(define iris (DataFrame-iris))
(format #t "iris: ~a rows x ~a cols\n\n" (DataFrame-nrows iris) (DataFrame-ncols iris))

(define model (new-GBMClassifier iris "Species ~ Petal.Length + Petal.Width"))
(GBMClassifier-print-summary model)

(format #t "\nConfusion matrix (rows = actual, cols = predicted):\n")
(format #t "~a\n" (DataFrame-to-string (GBMClassifier-confusion-matrix model)))

(format #t "\nMisclassified rows:\n")
(define predictions (GBMClassifier-predict model iris))
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

(Plot-save (GBMClassifier-plot-training-deviance model) "gbm_iris_training_deviance.svg")
(Plot-save (GBMClassifier-plot-decision-regions model "Petal.Length" "Petal.Width") "gbm_iris_decision_regions.svg")
(format #t "\nSaved gbm_iris_training_deviance.svg and gbm_iris_decision_regions.svg\n")

(define few (new-GBMClassifier iris "Species ~ Petal.Length + Petal.Width" 5))
(define few-dev (GBMClassifier-training-deviance few))
(define model-dev (GBMClassifier-training-deviance model))
(format #t "\n5-round ensemble:   training accuracy=~a%  deviance=~a\n" (* 100.0 (GBMClassifier-training-accuracy few)) (vector-ref few-dev (- (vector-length few-dev) 1)))
(format #t "100-round ensemble: training accuracy=~a%  deviance=~a\n" (* 100.0 (GBMClassifier-training-accuracy model)) (vector-ref model-dev (- (vector-length model-dev) 1)))
(Plot-save (GBMClassifier-plot-decision-regions few "Petal.Length" "Petal.Width") "gbm_iris_decision_regions_5rounds.svg")
(format #t "Saved gbm_iris_decision_regions_5rounds.svg\n")
