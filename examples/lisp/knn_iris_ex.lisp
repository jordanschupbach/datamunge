;; CMD: just TARGET=knn_iris_ex run-lisp

;;;; k-nearest-neighbours classifier on iris: leave-one-out accuracy and decision regions for several k.
(require "asdf")
(asdf:load-system "datamunge")
(setf *read-default-float-format* (quote double-float)) ; print doubles without a d0 suffix

(defparameter *iris* (dm.data:iris))
(format t "iris: ~a rows x ~a cols~%~%" (dm:nrows *iris*) (dm:ncols *iris*))

(defparameter *model* (dm:make-knn-classifier *iris* "Species ~ Petal.Length + Petal.Width"))
(dm:print-summary *model*)

(format t "~%Confusion matrix (leave-one-out; rows = actual, cols = predicted):~%")
(format t "~a~%" (dm:to-string (dm:confusion-matrix *model*)))

(format t "~%Leave-one-out misclassified rows:~%")
(let* ((fitted (dm:predict *model* *iris*))
       (n (dm:nrows *iris*))
       (misclassified 0))
  (dotimes (i n)
    (let ((actual (dm:string-at *iris* "Species" i))
          (predicted (aref fitted i)))
      (unless (string= predicted actual)
        (incf misclassified)
        (format t "  row ~a: Petal.Length=~a Petal.Width=~a  actual=~a  predicted=~a~%"
                i (dm:numeric-at *iris* "Petal.Length" i) (dm:numeric-at *iris* "Petal.Width" i)
                actual predicted))))
  (format t "~a of ~a misclassified (~a%)~%" misclassified n (/ (* 100d0 misclassified) n)))

(dm:save (dm:plot-decision-regions *model* "Petal.Length" "Petal.Width")
         "knn_iris_decision_regions_k5.svg")
(format t "~%Saved knn_iris_decision_regions_k5.svg~%")

;;; k=1 memorizes every training point exactly (jagged, overfit boundary with an island around
;;; every point, including noise); k=25 averages over a much larger neighborhood (very smooth,
;;; underfit boundary). (make-knn-classifier data formula &optional k metric weighted standardize)
(dolist (k '(1 25))
  (let ((knn (dm:make-knn-classifier *iris* "Species ~ Petal.Length + Petal.Width" k))
        (file (format nil "knn_iris_decision_regions_k~a.svg" k)))
    (format t "~%k=~2a leave-one-out accuracy: ~a%~%" k (* (dm:training-accuracy knn) 100d0))
    (dm:save (dm:plot-decision-regions knn "Petal.Length" "Petal.Width") file)
    (format t "Saved ~a~%" file)))
