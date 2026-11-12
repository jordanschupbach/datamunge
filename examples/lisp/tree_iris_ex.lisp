;;;; Decision tree classifier on iris: summary, confusion matrix, misclassified rows and plots.
(require "asdf")
(asdf:load-system "datamunge")

(setf *read-default-float-format* 'double-float)

(defparameter *iris* (dm:data-frame-iris))
(format t "iris: ~a rows x ~a cols~%~%" (dm:nrows *iris*) (dm:ncols *iris*))

(defparameter *model* (dm:make-decision-tree-classifier *iris* "Species ~ Petal.Length + Petal.Width"))
(dm:print-summary *model*)

(format t "~%Confusion matrix (rows = actual, cols = predicted):~%")
(format t "~a~%" (dm:to-string (dm:confusion-matrix *model*)))

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

(dm:save (dm:plot-classification *model* *iris* "Petal.Length" "Petal.Width") "tree_iris_classification.svg")
(dm:save (dm:plot-decision-regions *model* "Petal.Length" "Petal.Width") "tree_iris_decision_regions.svg")
(format t "~%Saved tree_iris_classification.svg and tree_iris_decision_regions.svg~%")

;; A shallower tree, for comparison, showing a coarser (but still fairly accurate) decision boundary.
(let ((shallow (dm:make-decision-tree-classifier *iris* "Species ~ Petal.Length + Petal.Width" 2)))
  (format t "~%Depth-2 tree training accuracy: ~a% (~a leaves)~%"
          (* (dm:training-accuracy shallow) 100d0) (dm:leaf-count shallow))
  (dm:save (dm:plot-decision-regions shallow "Petal.Length" "Petal.Width") "tree_iris_decision_regions_depth2.svg")
  (format t "Saved tree_iris_decision_regions_depth2.svg~%"))
