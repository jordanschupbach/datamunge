;;;; Linear model: mpg ~ hp + wt + transmission, its ANOVA table and predictions.
(require "asdf")
(asdf:load-system "datamunge")

(defparameter *cars*
  (let ((frame (dm:make-data-frame)))
    (dm:add-numeric-column frame "hp" #(110d0 110d0 93d0 110d0 175d0 105d0 245d0 62d0 95d0 123d0))
    (dm:add-numeric-column frame "wt" #(2.62d0 2.875d0 2.32d0 3.215d0 3.44d0 3.46d0 3.57d0 3.19d0 3.15d0 3.44d0))
    (dm:add-string-column frame "transmission"
                          #("manual" "manual" "manual" "automatic" "automatic"
                            "automatic" "automatic" "automatic" "automatic" "automatic"))
    (dm:add-numeric-column frame "mpg" #(21d0 21d0 22.8d0 21.4d0 18.7d0 18.1d0 14.3d0 24.4d0 22.8d0 19.2d0))
    frame))

(format t "Fitting: mpg ~~ hp + wt + transmission~%~%")
(defparameter *model* (dm:make-lm *cars* "mpg ~ hp + wt + transmission"))
(dm:print-summary *model*)

(format t "~%Sequential ANOVA:~%~a~%" (dm:to-string (dm:anova *model*)))

(let ((newcars (dm:make-data-frame)))
  (dm:add-numeric-column newcars "hp" #(150d0 90d0))
  (dm:add-numeric-column newcars "wt" #(3d0 2.5d0))
  (dm:add-string-column newcars "transmission" #("manual" "automatic"))
  (format t "~%Predictions with 95% confidence intervals:~%~a~%"
          (dm:to-string (dm:predict-frame *model* newcars "confidence"))))

(dm:save-diagnostic-plots *model* "lm_ex_diagnostics")
(format t "~%Saved diagnostic plots as lm_ex_diagnostics_*.svg~%")
