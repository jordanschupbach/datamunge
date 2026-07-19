(use-modules (datamunge))

(define FORMULA "Petal.Length ~ Petal.Width")

(define iris (DataFrame-iris))
(format #t "iris: ~a rows x ~a cols\n" (DataFrame-nrows iris) (DataFrame-ncols iris))
(format #t "formula: ~a\n\n" FORMULA)

(define model (new-GaussianProcessRegression iris FORMULA))
(GaussianProcessRegression-print-summary model)

(Plot-save (GaussianProcessRegression-plot-fit model iris) "gpr_iris_fit.svg")
(Plot-save (GaussianProcessRegression-plot-length-scale-profile model) "gpr_iris_length_scale_profile.svg")
(format #t "\nSaved gpr_iris_fit.svg and gpr_iris_length_scale_profile.svg\n")

(define query (DataFrame-empty))
(DataFrame-add-numeric-column query "Petal.Width" (list 0.2 1.3 2.5 10.0))
(define detail (GaussianProcessRegression-predict-frame model query "confidence"))
(format #t "\nPredictions with 95% confidence intervals:\n")
(format #t "~a\n" (DataFrame-to-string detail))
(format #t "(Petal.Width=10.0 is far outside the training range [0.1, 2.5] -- note how much wider its\n interval is than the in-range predictions.)\n")
