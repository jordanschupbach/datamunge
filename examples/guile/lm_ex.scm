(use-modules (datamunge))

(define hp (list 110.0 110.0 93.0 110.0 175.0 105.0 245.0 62.0 95.0 123.0))
(define wt (list 2.62 2.875 2.32 3.215 3.44 3.46 3.57 3.19 3.15 3.44))
(define transmission (list "manual" "manual" "manual" "automatic" "automatic" "automatic" "automatic" "automatic" "automatic" "automatic"))
(define mpg (list 21.0 21.0 22.8 21.4 18.7 18.1 14.3 24.4 22.8 19.2))

(define cars (DataFrame-empty))
(DataFrame-add-numeric-column cars "hp" hp)
(DataFrame-add-numeric-column cars "wt" wt)
(DataFrame-add-string-column cars "transmission" transmission)
(DataFrame-add-numeric-column cars "mpg" mpg)

(format #t "Fitting: mpg ~~ hp + wt + transmission\n\n")
(define model (new-LM cars "mpg ~ hp + wt + transmission"))
(LM-print-summary model)

(format #t "\nSequential ANOVA:\n")
(format #t "~a\n" (DataFrame-to-string (LM-anova model)))

(define newcars (DataFrame-empty))
(DataFrame-add-numeric-column newcars "hp" (list 150.0 90.0))
(DataFrame-add-numeric-column newcars "wt" (list 3.0 2.5))
(DataFrame-add-string-column newcars "transmission" (list "manual" "automatic"))

(define frame (LM-predict-frame model newcars "confidence"))
(format #t "\nPredictions with 95% confidence intervals:\n")
(format #t "~a\n" (DataFrame-to-string frame))

(LM-save-diagnostic-plots model "lm_ex_diagnostics")
(format #t "\nSaved diagnostic plots as lm_ex_diagnostics_*.svg\n")
