;; CMD: just TARGET=gaussian_process_regression_iris_ex run-guile
;; Gaussian process regression of Petal.Length on Petal.Width (iris), with confidence intervals.
(use-modules (oop goops) (datamunge) ((datamunge data) #:select (iris)))

(define formula "Petal.Length ~ Petal.Width")

(define iris-df (iris))
(format #t "iris: ~a rows x ~a cols\n" (nrows iris-df) (ncols iris-df))
(format #t "formula: ~a\n\n" formula)

;; Both the length scale and the noise ratio are auto-selected by maximizing the exact log
;; marginal likelihood.
(define model (make <gaussian-process-regression> iris-df formula))
(print-summary model)

(save (plot-fit model iris-df) "gpr_iris_fit.svg")
(save (plot-length-scale-profile model) "gpr_iris_length_scale_profile.svg")
(display "\nSaved gpr_iris_fit.svg and gpr_iris_length_scale_profile.svg\n")

;; Unlike every other regressor in this suite, a GP gives a genuine posterior confidence
;; interval at every point -- including far outside the training data, where it should widen
;; substantially as the model's uncertainty grows.
(define query (make <data-frame>))
(add-numeric-column query "Petal.Width" '(0.2 1.3 2.5 10.0))
(define detail (predict-frame model query "confidence"))
(display "\nPredictions with 95% confidence intervals:\n")
(format #t "~a\n" (to-string detail))
(display "(Petal.Width=10.0 is far outside the training range [0.1, 2.5] -- note how much wider its\n interval is than the in-range predictions.)\n")
