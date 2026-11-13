;; CMD: just TARGET=gaussian_process_regression_iris_ex run-lisp

;;;; Gaussian process regression of Petal.Length on Petal.Width (iris), with confidence intervals.
(require "asdf")
(asdf:load-system "datamunge")

(defparameter +formula+ "Petal.Length ~ Petal.Width")

(defparameter *iris* (dm.data:iris))
(format t "iris: ~a rows x ~a cols~%" (dm:nrows *iris*) (dm:ncols *iris*))
(format t "formula: ~a~%~%" +formula+)

;; Both the length scale and the noise ratio are auto-selected by maximizing the exact log
;; marginal likelihood.
(defparameter *model* (dm:make-gaussian-process-regression *iris* +formula+))
(dm:print-summary *model*)

(dm:save (dm:plot-fit *model* *iris*) "gpr_iris_fit.svg")
(dm:save (dm:plot-length-scale-profile *model*) "gpr_iris_length_scale_profile.svg")
(format t "~%Saved gpr_iris_fit.svg and gpr_iris_length_scale_profile.svg~%")

;; Unlike every other regressor in this suite, a GP gives a genuine posterior confidence
;; interval at every point -- including far outside the training data, where it should widen
;; substantially as the model's uncertainty grows.
(let ((query (dm:make-data-frame)))
  (dm:add-numeric-column query "Petal.Width" #(0.2d0 1.3d0 2.5d0 10d0))
  (let ((detail (dm:predict-frame *model* query "confidence")))
    (format t "~%Predictions with 95% confidence intervals:~%")
    (format t "~a~%" (dm:to-string detail))
    (format t "(Petal.Width=10.0 is far outside the training range [0.1, 2.5] -- note how much wider its~% interval is than the in-range predictions.)~%")))
