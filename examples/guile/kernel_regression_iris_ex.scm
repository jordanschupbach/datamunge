;; CMD: just TARGET=kernel_regression_iris_ex run-guile
;; Nadaraya-Watson kernel regression on iris: CV-selected bandwidth vs. too-small and too-large ones.
(use-modules (oop goops) (datamunge) ((datamunge data) #:select (iris)))

(define formula "Petal.Length ~ Petal.Width")

(define iris-df (iris))
(format #t "iris: ~a rows x ~a cols\n" (nrows iris-df) (ncols iris-df))
(format #t "formula: ~a\n\n" formula)

(define model (make <kernel-regression> iris-df formula)) ; bandwidth auto-selected via leave-one-out CV
(print-summary model)

(save (plot-fit model iris-df) "kernel_regression_iris_fit.svg")
(save (plot-cv-curve model) "kernel_regression_iris_cv.svg")
(display "\nSaved kernel_regression_iris_fit.svg and kernel_regression_iris_cv.svg\n")

;; A too-small and a too-large bandwidth, for comparison -- the classic kernel regression
;; bias-variance story.
(define small (make <kernel-regression> iris-df formula #:default 0.05))
(format #t "\nbandwidth=0.05 (too small): LOO R-squared=~a  LOO RMSE=~a\n" (r-squared small) (rmse small))
(save (plot-fit small iris-df) "kernel_regression_iris_fit_small_bandwidth.svg")

(define large (make <kernel-regression> iris-df formula #:default 5.0))
(format #t "bandwidth=5.0 (too large):  LOO R-squared=~a  LOO RMSE=~a\n" (r-squared large) (rmse large))
(save (plot-fit large iris-df) "kernel_regression_iris_fit_large_bandwidth.svg")

(format #t "bandwidth=~a (CV-selected): LOO R-squared=~a  LOO RMSE=~a\n"
        (bandwidth model) (r-squared model) (rmse model))
(display "\nSaved kernel_regression_iris_fit_small_bandwidth.svg and kernel_regression_iris_fit_large_bandwidth.svg\n")
