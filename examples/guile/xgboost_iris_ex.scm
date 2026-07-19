(use-modules (datamunge))

(define iris (DataFrame-iris))
(format #t "iris: ~a rows x ~a cols\n\n" (DataFrame-nrows iris) (DataFrame-ncols iris))

(define model (new-XGBoostClassifier iris "Species ~ Petal.Length + Petal.Width"))
(XGBoostClassifier-print-summary model)

(format #t "\nConfusion matrix (rows = actual, cols = predicted):\n")
(format #t "~a\n" (DataFrame-to-string (XGBoostClassifier-confusion-matrix model)))

(format #t "\nMisclassified rows:\n")
(define predictions (XGBoostClassifier-predict model iris))
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

(Plot-save (XGBoostClassifier-plot-training-deviance model) "xgboost_iris_training_deviance.svg")
(Plot-save (XGBoostClassifier-plot-decision-regions model "Petal.Length" "Petal.Width") "xgboost_iris_decision_regions.svg")
(format #t "\nSaved xgboost_iris_training_deviance.svg and xgboost_iris_decision_regions.svg\n")

(define heavy (new-XGBoostClassifier iris "Species ~ Petal.Length + Petal.Width" 100 0.3 6 50.0))
(define model-dev (XGBoostClassifier-training-deviance model))
(define heavy-dev (XGBoostClassifier-training-deviance heavy))
(format #t "\nlambda=1 (default):   training accuracy=~a%  deviance=~a\n" (* 100.0 (XGBoostClassifier-training-accuracy model)) (vector-ref model-dev (- (vector-length model-dev) 1)))
(format #t "lambda=50 (heavy L2): training accuracy=~a%  deviance=~a\n" (* 100.0 (XGBoostClassifier-training-accuracy heavy)) (vector-ref heavy-dev (- (vector-length heavy-dev) 1)))
(Plot-save (XGBoostClassifier-plot-decision-regions heavy "Petal.Length" "Petal.Width") "xgboost_iris_decision_regions_heavy_lambda.svg")
(format #t "Saved xgboost_iris_decision_regions_heavy_lambda.svg\n")
