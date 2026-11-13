;; CMD: just TARGET=forest_iris_ex run-lisp

;;;; Random forest classifier on iris: summary, confusion matrix, misclassifications, OOB accuracy.
(require "asdf")
(asdf:load-system "datamunge")
(setf *read-default-float-format* (quote double-float)) ; print doubles without a d0 suffix

(defparameter *iris* (dm.data:iris))
(format t "iris: ~a rows x ~a cols~%~%" (dm:nrows *iris*) (dm:ncols *iris*))

(defparameter *model* (dm:make-random-forest-classifier *iris* "Species ~ Petal.Length + Petal.Width"))
(dm:print-summary *model*)

(format t "~%Confusion matrix (rows = actual, cols = predicted):~%")
(format t "~a~%" (dm:to-string (dm:confusion-matrix *model*)))

(format t "~%Misclassified rows:~%")
(let* ((predictions (dm:predict *model* *iris*))
       (n (dm:nrows *iris*))
       (misclassified 0))
  (dotimes (i n)
    (let ((actual (dm:string-at *iris* "Species" i))
          (predicted (aref predictions i)))
      (unless (string= predicted actual)
        (incf misclassified)
        (format t "  row ~a: Petal.Length=~a Petal.Width=~a  actual=~a  predicted=~a~%"
                i (dm:numeric-at *iris* "Petal.Length" i) (dm:numeric-at *iris* "Petal.Width" i)
                actual predicted))))
  (format t "~a of ~a misclassified (~a%)~%" misclassified n (/ (* 100d0 misclassified) n)))

(dm:save (dm:plot-classification *model* *iris* "Petal.Length" "Petal.Width")
         "forest_iris_classification.svg")
(dm:save (dm:plot-decision-regions *model* "Petal.Length" "Petal.Width")
         "forest_iris_decision_regions.svg")
(format t "~%Saved forest_iris_classification.svg and forest_iris_decision_regions.svg~%")

;;; A small forest, for comparison, showing how out-of-bag accuracy stabilizes as more trees
;;; are added -- the defining random forest effect a single decision tree cannot demonstrate.
;;; (make-random-forest-classifier data formula &optional n-trees max-depth min-samples-split
;;;   min-samples-leaf max-features criterion bootstrap sample-fraction seed)
(let ((small-forest (dm:make-random-forest-classifier *iris* "Species ~ Petal.Length + Petal.Width" 5)))
  (format t "~%5-tree forest:   training accuracy=~a%  OOB accuracy=~a%~%"
          (* (dm:training-accuracy small-forest) 100d0) (* (dm:oob-accuracy small-forest) 100d0))
  (format t "100-tree forest: training accuracy=~a%  OOB accuracy=~a%~%"
          (* (dm:training-accuracy *model*) 100d0) (* (dm:oob-accuracy *model*) 100d0))
  (dm:save (dm:plot-decision-regions small-forest "Petal.Length" "Petal.Width")
           "forest_iris_decision_regions_5trees.svg")
  (format t "Saved forest_iris_decision_regions_5trees.svg~%"))
