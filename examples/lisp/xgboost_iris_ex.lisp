;; CMD: just TARGET=xgboost_iris_ex run-lisp

;;;; XGBoost classifier on iris: summary, confusion matrix, misclassified rows, L2-regularization comparison.
(require "asdf")
(asdf:load-system "datamunge")

;; Print doubles as 5.1 rather than 5.1d0.
(setf *read-default-float-format* 'double-float)

(defparameter *iris* (dm.data:iris))
(format t "iris: ~a rows x ~a cols~%~%" (dm:nrows *iris*) (dm:ncols *iris*))

(defparameter *model* (dm:make-xg-boost-classifier *iris* "Species ~ Petal.Length + Petal.Width"))
(dm:print-summary *model*)

(format t "~%Confusion matrix (rows = actual, cols = predicted):~%~a~%"
        (dm:to-string (dm:confusion-matrix *model*)))

(format t "~%Misclassified rows:~%")
(let ((predictions (dm:predict *model* *iris*))
      (misclassified 0)
      (n (dm:nrows *iris*)))
  (dotimes (i n)
    (let ((actual (dm:string-at *iris* "Species" i))
          (predicted (aref predictions i)))
      (unless (string= predicted actual)
        (incf misclassified)
        (format t "  row ~a: Petal.Length=~a Petal.Width=~a  actual=~a  predicted=~a~%"
                i (dm:numeric-at *iris* "Petal.Length" i) (dm:numeric-at *iris* "Petal.Width" i)
                actual predicted))))
  (format t "~a of ~a misclassified (~a%)~%" misclassified n (/ (* 100d0 misclassified) n)))

(dm:save (dm:plot-training-deviance *model*) "xgboost_iris_training_deviance.svg")
(dm:save (dm:plot-decision-regions *model* "Petal.Length" "Petal.Width") "xgboost_iris_decision_regions.svg")
(format t "~%Saved xgboost_iris_training_deviance.svg and xgboost_iris_decision_regions.svg~%")

;; XGBoost's defining lever is regularization: a heavily L2-regularized model (large lambda)
;; grows the same deep trees but keeps every leaf weight small, producing a much smoother
;; decision boundary than the lightly-regularized default.
;; (make-xg-boost-classifier data formula &optional n-trees learning-rate max-depth lambda alpha
;;                           gamma min-child-weight min-samples-leaf subsample colsample-bytree seed)
(let* ((heavy (dm:make-xg-boost-classifier *iris* "Species ~ Petal.Length + Petal.Width" 100 0.3d0 6 50d0))
       (model-dev (dm:training-deviance *model*))
       (heavy-dev (dm:training-deviance heavy)))
  (format t "~%lambda=1 (default):   training accuracy=~a%  deviance=~a~%"
          (* 100d0 (dm:training-accuracy *model*)) (aref model-dev (1- (length model-dev))))
  (format t "lambda=50 (heavy L2): training accuracy=~a%  deviance=~a~%"
          (* 100d0 (dm:training-accuracy heavy)) (aref heavy-dev (1- (length heavy-dev))))
  (dm:save (dm:plot-decision-regions heavy "Petal.Length" "Petal.Width")
           "xgboost_iris_decision_regions_heavy_lambda.svg")
  (format t "Saved xgboost_iris_decision_regions_heavy_lambda.svg~%"))
