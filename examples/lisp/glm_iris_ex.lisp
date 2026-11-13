;; CMD: just TARGET=glm_iris_ex run-lisp

;;;; Generalized linear models on iris: logistic regression (binomial) and Poisson regression.
(require "asdf")
(asdf:load-system "datamunge")
(setf *read-default-float-format* (quote double-float)) ; print doubles without a d0 suffix

(defparameter *iris* (dm.data:iris))

;;; Logistic regression: versicolor vs virginica only -- setosa is perfectly separable from
;;; the other two on these predictors, which sends logistic regression's coefficients toward
;;; +/-infinity (a genuine degeneracy of the method, not a bug) -- so this is the well-behaved
;;; binary split for a demo.
(defparameter *is-virginica* nil)
(defparameter *petal-length* nil)
(defparameter *petal-width* nil)
(dotimes (i (dm:nrows *iris*))
  (let ((species (dm:string-at *iris* "Species" i)))
    (when (or (string= species "versicolor") (string= species "virginica"))
      (push (if (string= species "virginica") 1d0 0d0) *is-virginica*)
      (push (dm:numeric-at *iris* "Petal.Length" i) *petal-length*)
      (push (dm:numeric-at *iris* "Petal.Width" i) *petal-width*))))
(setf *is-virginica* (coerce (nreverse *is-virginica*) 'vector)
      *petal-length* (coerce (nreverse *petal-length*) 'vector)
      *petal-width* (coerce (nreverse *petal-width*) 'vector))

(defparameter *sub*
  (let ((frame (dm:make-data-frame)))
    (dm:add-numeric-column frame "Petal.Length" *petal-length*)
    (dm:add-numeric-column frame "Petal.Width" *petal-width*)
    (dm:add-numeric-column frame "is_virginica" *is-virginica*)
    frame))

(format t "=================== Logistic regression (binomial, logit link) ===================~%")
(defparameter *logit* (dm:make-glm *sub* "is_virginica ~ Petal.Length + Petal.Width" "binomial"))
(dm:print-summary *logit*)

(let* ((fitted (dm:fitted-values *logit*))
       (n (length *is-virginica*))
       (correct (loop for i below n
                      count (eq (>= (aref fitted i) 0.5d0) (>= (aref *is-virginica* i) 0.5d0)))))
  (format t "~%Resubstitution accuracy at 0.5 threshold: ~a%~%" (/ (* 100d0 correct) n)))

(dm:save-diagnostic-plots *logit* "glm_logistic_iris")
(format t "~%Saved glm_logistic_iris_{residuals_vs_fitted,normal_qq,scale_location,residuals_vs_leverage}.svg~%")

;;; Predicted-probability curve across Petal.Length, with Petal.Width held at its mean -- the
;;; classic sigmoid shape of a fitted logistic regression, with a 95% confidence band.
(let* ((width-mean (/ (reduce #'+ *petal-width*) (length *petal-width*)))
       (grid-n 100)
       (pl-min (- (reduce #'min *petal-length*) 0.3d0))
       (pl-max (+ (reduce #'max *petal-length*) 0.3d0))
       (grid-x (loop for i below grid-n
                     collect (+ pl-min (/ (* (- pl-max pl-min) i) (1- grid-n)))))
       (grid (dm:make-data-frame)))
  (dm:add-numeric-column grid "Petal.Length" grid-x)
  (dm:add-numeric-column grid "Petal.Width" (make-list grid-n :initial-element width-mean))
  (let ((curve-frame (dm:predict-frame *logit* grid "confidence")))
    (format t "~%Predicted-probability curve (first 5 rows):~%")
    (format t "~a~%" (dm:to-string curve-frame 5))))

;;; Poisson regression, for contrast: same IRLS engine, different family/link.
(format t "~%=================== Poisson regression (log link) ===================~%")
(let* ((n (dm:nrows *iris*))
       (count (loop for i below n
                    collect (float (floor (+ (dm:numeric-at *iris* "Sepal.Length" i) 0.5d0)) 1d0)))
       (sepal-width (loop for i below n collect (dm:numeric-at *iris* "Sepal.Width" i)))
       (all-petal-length (loop for i below n collect (dm:numeric-at *iris* "Petal.Length" i)))
       (count-data (dm:make-data-frame)))
  (dm:add-numeric-column count-data "Sepal.Width" sepal-width)
  (dm:add-numeric-column count-data "Petal.Length" all-petal-length)
  (dm:add-numeric-column count-data "count" count)
  (dm:print-summary (dm:make-glm count-data "count ~ Sepal.Width + Petal.Length" "poisson")))
