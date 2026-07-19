(use-modules (datamunge))

(define FORMULA "Petal.Length ~ Petal.Width")

(define iris (DataFrame-iris))
(format #t "iris: ~a rows x ~a cols\n" (DataFrame-nrows iris) (DataFrame-ncols iris))
(format #t "formula: ~a\n\n" FORMULA)

(define model (new-KernelRegression iris FORMULA))
(KernelRegression-print-summary model)

(Plot-save (KernelRegression-plot-fit model iris) "kernel_regression_iris_fit.svg")
(Plot-save (KernelRegression-plot-cv-curve model) "kernel_regression_iris_cv.svg")
(format #t "\nSaved kernel_regression_iris_fit.svg and kernel_regression_iris_cv.svg\n")

(define small (new-KernelRegression iris FORMULA "gaussian" 0.05))
(format #t "\nbandwidth=0.05 (too small): LOO R-squared=~a  LOO RMSE=~a\n" (KernelRegression-r-squared small) (KernelRegression-rmse small))
(Plot-save (KernelRegression-plot-fit small iris) "kernel_regression_iris_fit_small_bandwidth.svg")

(define large (new-KernelRegression iris FORMULA "gaussian" 5.0))
(format #t "bandwidth=5.0 (too large):  LOO R-squared=~a  LOO RMSE=~a\n" (KernelRegression-r-squared large) (KernelRegression-rmse large))
(Plot-save (KernelRegression-plot-fit large iris) "kernel_regression_iris_fit_large_bandwidth.svg")

(format #t "bandwidth=~a (CV-selected): LOO R-squared=~a  LOO RMSE=~a\n" (KernelRegression-bandwidth model) (KernelRegression-r-squared model) (KernelRegression-rmse model))
(format #t "\nSaved kernel_regression_iris_fit_small_bandwidth.svg and kernel_regression_iris_fit_large_bandwidth.svg\n")
