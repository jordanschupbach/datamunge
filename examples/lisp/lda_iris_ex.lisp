;; CMD: just TARGET=lda_iris_ex run-lisp

;;;; Linear discriminant analysis on iris: summary, confusion matrix, predictions and discriminant plot.
(require "asdf")
(asdf:load-system "datamunge")

(defparameter *iris* (dm.data:iris))
(format t "iris: ~a rows x ~a cols~%~%" (dm:nrows *iris*) (dm:ncols *iris*))

(defparameter *model*
  (dm:make-lda *iris* "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width"))
(dm:print-summary *model*)

(format t "~%Confusion matrix (rows = actual, cols = predicted):~%~a~%"
        (dm:to-string (dm:confusion-matrix *model*)))

(let ((newdata (dm:make-data-frame)))
  (dm:add-numeric-column newdata "Sepal.Length" #(5.1d0 6.0d0 6.5d0 6.2d0))
  (dm:add-numeric-column newdata "Sepal.Width" #(3.5d0 2.7d0 3.0d0 2.8d0))
  (dm:add-numeric-column newdata "Petal.Length" #(1.4d0 4.5d0 5.5d0 4.8d0))
  (dm:add-numeric-column newdata "Petal.Width" #(0.2d0 1.5d0 2.0d0 1.8d0))
  (format t "~%Predictions for new flowers:~%~a~%" (dm:to-string (dm:predict-frame *model* newdata))))

(dm:save-discriminant-plot *model* "lda_iris_discriminants.svg")
(format t "~%Saved discriminant plot as lda_iris_discriminants.svg~%")
