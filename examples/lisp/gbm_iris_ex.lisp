;;;; Gradient-boosted trees (GBM) on iris: summary, confusion matrix, misclassified rows, 5 vs 100 rounds.
(require "asdf")
(asdf:load-system "datamunge")

;; Print doubles as 5.1 rather than 5.1d0.
(setf *read-default-float-format* 'double-float)

(defparameter *iris* (dm:data-frame-iris))
(format t "iris: ~a rows x ~a cols~%~%" (dm:nrows *iris*) (dm:ncols *iris*))

(defparameter *model* (dm:make-gbm-classifier *iris* "Species ~ Petal.Length + Petal.Width"))
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

(dm:save (dm:plot-training-deviance *model*) "gbm_iris_training_deviance.svg")
(dm:save (dm:plot-decision-regions *model* "Petal.Length" "Petal.Width") "gbm_iris_decision_regions.svg")
(format t "~%Saved gbm_iris_training_deviance.svg and gbm_iris_decision_regions.svg~%")

;; A handful of boosting rounds vs a well-boosted ensemble: each additional round chips away
;; at the training loss, gradually sharpening the decision boundary.
;; (make-gbm-classifier data formula &optional n-trees learning-rate max-depth min-samples-split
;;                      min-samples-leaf subsample seed)
(let* ((few (dm:make-gbm-classifier *iris* "Species ~ Petal.Length + Petal.Width" 5))
       (few-dev (dm:training-deviance few))
       (model-dev (dm:training-deviance *model*)))
  (format t "~%5-round ensemble:   training accuracy=~a%  deviance=~a~%"
          (* 100d0 (dm:training-accuracy few)) (aref few-dev (1- (length few-dev))))
  (format t "100-round ensemble: training accuracy=~a%  deviance=~a~%"
          (* 100d0 (dm:training-accuracy *model*)) (aref model-dev (1- (length model-dev))))
  (dm:save (dm:plot-decision-regions few "Petal.Length" "Petal.Width") "gbm_iris_decision_regions_5rounds.svg")
  (format t "Saved gbm_iris_decision_regions_5rounds.svg~%"))
