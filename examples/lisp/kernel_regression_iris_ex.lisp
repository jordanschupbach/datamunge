;; CMD: just TARGET=kernel_regression_iris_ex run-lisp

;;;; Nadaraya-Watson kernel regression on iris: CV-selected bandwidth vs. too-small and too-large ones.
(require "asdf")
(asdf:load-system "datamunge")

(defparameter *formula* "Petal.Length ~ Petal.Width")

(defparameter *iris* (dm.data:iris))
(format t "iris: ~a rows x ~a cols~%" (dm:nrows *iris*) (dm:ncols *iris*))
(format t "formula: ~a~%~%" *formula*)

(defparameter *model* (dm:make-kernel-regression *iris* *formula*)) ; bandwidth auto-selected via leave-one-out CV
(dm:print-summary *model*)

(dm:save (dm:plot-fit *model* *iris*) "kernel_regression_iris_fit.svg")
(dm:save (dm:plot-cv-curve *model*) "kernel_regression_iris_cv.svg")
(format t "~%Saved kernel_regression_iris_fit.svg and kernel_regression_iris_cv.svg~%")

;; A too-small and a too-large bandwidth, for comparison -- the classic kernel regression
;; bias-variance story. (make-kernel-regression data formula kernel bandwidth n-bandwidth standardize)
(let ((small (dm:make-kernel-regression *iris* *formula* "gaussian" 0.05d0)))
  (format t "~%bandwidth=0.05 (too small): LOO R-squared=~f  LOO RMSE=~f~%"
          (dm:r-squared small) (dm:rmse small))
  (dm:save (dm:plot-fit small *iris*) "kernel_regression_iris_fit_small_bandwidth.svg"))

(let ((large (dm:make-kernel-regression *iris* *formula* "gaussian" 5d0)))
  (format t "bandwidth=5.0 (too large):  LOO R-squared=~f  LOO RMSE=~f~%"
          (dm:r-squared large) (dm:rmse large))
  (dm:save (dm:plot-fit large *iris*) "kernel_regression_iris_fit_large_bandwidth.svg"))

(format t "bandwidth=~f (CV-selected): LOO R-squared=~f  LOO RMSE=~f~%"
        (dm:bandwidth *model*) (dm:r-squared *model*) (dm:rmse *model*))
(format t "~%Saved kernel_regression_iris_fit_small_bandwidth.svg and ~
           kernel_regression_iris_fit_large_bandwidth.svg~%")
