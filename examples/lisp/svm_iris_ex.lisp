;; CMD: just TARGET=svm_iris_ex run-lisp

;;;; Support vector machine on iris: RBF vs. linear kernels, confusion matrix and predictions.
(require "asdf")
(asdf:load-system "datamunge")

(defparameter *formula* "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width")
(defparameter *iris* (dm.data:iris))
(format t "iris: ~a rows x ~a cols~%~%" (dm:nrows *iris*) (dm:ncols *iris*))

(format t "=== RBF kernel (default) ===~%")
(defparameter *rbf-model* (dm:make-svm *iris* *formula*))
(dm:print-summary *rbf-model*)
(format t "~%Confusion matrix (rows = actual, cols = predicted):~%~a~%"
        (dm:to-string (dm:confusion-matrix *rbf-model*)))

(format t "~%=== Linear kernel, for comparison ===~%")
(let ((linear-model (dm:make-svm *iris* *formula* "linear")))
  (format t "Training accuracy: ~f%~%" (* (dm:training-accuracy linear-model) 100d0))
  (format t "Support vectors: ~a~%" (dm:num-support-vectors linear-model)))

(let ((newdata (dm:make-data-frame)))
  (dm:add-numeric-column newdata "Sepal.Length" #(5.1d0 6.0d0 6.5d0 6.2d0))
  (dm:add-numeric-column newdata "Sepal.Width" #(3.5d0 2.7d0 3.0d0 2.8d0))
  (dm:add-numeric-column newdata "Petal.Length" #(1.4d0 4.5d0 5.5d0 4.8d0))
  (dm:add-numeric-column newdata "Petal.Width" #(0.2d0 1.5d0 2.0d0 1.8d0))
  (format t "~%RBF predictions for new flowers (votes out of 3 one-vs-one pairs):~%~a~%"
          (dm:to-string (dm:predict-frame *rbf-model* newdata))))
