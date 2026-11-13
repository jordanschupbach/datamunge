;; CMD: just TARGET=lmm_ex run-lisp

;;;; Linear mixed models: random intercept (penguins) and random intercept + slope (simulated schools).
(require "asdf")
(asdf:load-system "datamunge")
(setf *read-default-float-format* 'double-float) ; print doubles without a d0 suffix

(defvar *rng* (sb-ext:seed-random-state 2024))

(defun uniform ()
  ;; (0, 1]: avoids log(0) in the Box-Muller transform.
  (- 1d0 (random 1d0 *rng*)))

(defun gauss ()
  (let ((u1 (uniform)) (u2 (uniform)))
    (* (sqrt (* -2d0 (log u1))) (cos (* 2d0 pi u2)))))

(format t "=================== Random intercept on a real dataset (penguins) ===================~%")
(let ((penguins (dm.data:penguins)))
  (dm:print-summary
   (dm:make-lmm penguins "body_mass_g ~ flipper_length_mm + bill_length_mm + (1 | species)")))

(format t "~%=================== Random intercept + slope on a simulated multi-school dataset ===================~%")
;; 30 schools, ~25 students each: score depends on study_hours, but both the baseline score
;; and the return on an extra hour of study vary by school -- the textbook case for a
;; random-intercept-and-slope model.
(defparameter *n-schools* 30)
(defparameter *school-intercept* (make-array *n-schools* :element-type 'double-float))
(defparameter *school-slope* (make-array *n-schools* :element-type 'double-float))
(dotimes (s *n-schools*)
  (setf (aref *school-intercept* s) (* (gauss) 6d0)
        (aref *school-slope* s) (* (gauss) 1.2d0)))

(defparameter *true-intercept* 60d0)
(defparameter *true-slope* 3d0)

(defparameter *df*
  (let ((school '()) (study-hours '()) (score '()))
    (dotimes (s *n-schools*)
      (let ((n-students (+ 15 (random 21 *rng*))))
        (dotimes (k n-students)
          (let* ((hours (* (random 1d0 *rng*) 10d0))
                 (noise (* (gauss) 4d0)))
            (push (float s 1d0) school)
            (push hours study-hours)
            (push (+ *true-intercept* (aref *school-intercept* s)
                     (* (+ *true-slope* (aref *school-slope* s)) hours)
                     noise)
                  score)))))
    (let ((frame (dm:make-data-frame)))
      (dm:add-numeric-column frame "school" (nreverse school))
      (dm:add-numeric-column frame "study_hours" (nreverse study-hours))
      (dm:add-numeric-column frame "score" (nreverse score))
      frame)))

(defparameter *model* (dm:make-lmm *df* "score ~ study_hours + (1 + study_hours | school)"))
(dm:print-summary *model*)

(format t "~%True generating values: intercept=~a, slope=~a, random-intercept SD=6.0, random-slope SD=1.2, residual SD=4.0~%"
        *true-intercept* *true-slope*)

(format t "~%--- Best Linear Unbiased Predictors (BLUPs) for a few schools ---~%")
(let ((group-labels (dm:group-labels *model*)))
  (dolist (idx '(0 1 2))
    (let ((re (dm:random-effects-for-group *model* idx)))
      (format t "school ~a: intercept shift=~a, slope shift=~a~%"
              (elt group-labels idx) (aref re 0) (aref re 1)))))

(format t "~%--- Prediction: population-level vs. school-adjusted ---~%")
(let ((newdata-population (dm:make-data-frame))
      (newdata-school0 (dm:make-data-frame)))
  (dm:add-numeric-column newdata-population "study_hours" #(5d0))
  (dm:add-numeric-column newdata-school0 "study_hours" #(5d0))
  (dm:add-numeric-column newdata-school0 "school" #(0d0))
  (format t "5 study hours, unseen school:      ~a (fixed effects only)~%"
          (aref (dm:predict *model* newdata-population) 0))
  (format t "5 study hours, school 0 (known):    ~a (fixed effects + school 0's BLUP)~%"
          (aref (dm:predict *model* newdata-school0) 0)))
