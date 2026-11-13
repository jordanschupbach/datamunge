;; CMD: just TARGET=glmm_ex run-lisp

;;;; Generalized linear mixed models: a binomial GLMM on penguins and a Poisson GLMM on simulated store data.
(require "asdf")
(asdf:load-system "datamunge")

(defvar *rng* (sb-ext:seed-random-state 4242))

(defun uniform () (random 1d0 *rng*))

(defun gauss ()
  (let ((u1 (- 1d0 (uniform))) (u2 (uniform)))   ; 1 - u keeps log() away from 0
    (* (sqrt (* -2d0 (log u1))) (cos (* 2d0 pi u2)))))

(format t "=================== Binomial (logistic) mixed model on a real dataset (penguins) ===================~%")
;; Predicting sex from body mass, grouped by island: unlike species (which is almost perfectly
;; confounded with island in this dataset -- Gentoo penguins occur on Biscoe island only), sex
;; is not tied to location, so this is a well-behaved fit.
(let ((penguins (dm.data:penguins))
      (is-male '()) (body-mass '()) (island '()))
  (dotimes (i (dm:nrows penguins))
    (unless (or (dm:is-null penguins "sex" i)
                (dm:is-null penguins "body_mass_g" i)
                (dm:is-null penguins "island" i))
      (push (if (string= (dm:string-at penguins "sex" i) "male") 1d0 0d0) is-male)
      (push (dm:numeric-at penguins "body_mass_g" i) body-mass)
      (push (dm:string-at penguins "island" i) island)))
  (let ((sex-df (dm:make-data-frame)))
    (dm:add-numeric-column sex-df "is_male" (reverse is-male))
    (dm:add-numeric-column sex-df "body_mass_g" (reverse body-mass))
    (dm:add-string-column sex-df "island" (reverse island))
    (dm:print-summary (dm:make-glmm sex-df "is_male ~ body_mass_g + (1 | island)" "binomial"))))

(format t "~%=================== Poisson mixed model on simulated multi-site count data ===================~%")
;; 25 stores, ~20 days each: daily visit counts depend on a promo intensity score, but both
;; the baseline traffic and the promo's effectiveness vary by store.
(defparameter *n-stores* 25)
(defparameter *store-effect*
  (let ((v (make-array *n-stores*)))
    (dotimes (s *n-stores* v) (setf (aref v s) (* (gauss) 0.4d0)))))
(defparameter *true-intercept* 2d0)
(defparameter *true-slope* 0.3d0)

(defparameter *store-model*
  (let ((store '()) (promo '()) (visits '()))
    (dotimes (s *n-stores*)
      (let ((n-days (+ 15 (random 11 *rng*))))
        (dotimes (day n-days)
          (let* ((promo-intensity (* (uniform) 3d0))
                 (lam (exp (+ *true-intercept* (aref *store-effect* s) (* *true-slope* promo-intensity))))
                 ;; Knuth's Poisson sampler.
                 (l-thresh (exp (- lam)))
                 (k 0)
                 (p 1d0))
            (loop do (incf k)
                     (setf p (* p (uniform)))
                  until (<= p l-thresh))
            (push (float s 1d0) store)
            (push promo-intensity promo)
            (push (float (1- k) 1d0) visits)))))
    (let ((df (dm:make-data-frame)))
      (dm:add-numeric-column df "store" (reverse store))
      (dm:add-numeric-column df "promo" (reverse promo))
      (dm:add-numeric-column df "visits" (reverse visits))
      (dm:make-glmm df "visits ~ promo + (1 | store)" "poisson"))))

(dm:print-summary *store-model*)

(format t "~%True generating values: intercept=~f, slope=~f, random-intercept SD (log scale)=0.4~%"
        *true-intercept* *true-slope*)

(format t "~%--- BLUPs for a few stores ---~%")
(let ((group-labels (dm:group-labels *store-model*)))
  (dolist (idx '(0 1 2))
    (format t "store ~a: intercept shift=~f~%"
            (elt group-labels idx)
            (elt (dm:random-effects-for-group *store-model* idx) 0))))

(format t "~%--- Prediction: population-level vs. store-adjusted ---~%")
(let ((newdata-population (dm:make-data-frame))
      (newdata-store0 (dm:make-data-frame)))
  (dm:add-numeric-column newdata-population "promo" #(1.5d0))
  (dm:add-numeric-column newdata-store0 "promo" #(1.5d0))
  (dm:add-numeric-column newdata-store0 "store" #(0d0))
  (format t "promo=1.5, unseen store:   ~f expected visits (fixed effects only)~%"
          (elt (dm:predict *store-model* newdata-population) 0))
  (format t "promo=1.5, store 0 (known): ~f expected visits (fixed effects + store 0's BLUP)~%"
          (elt (dm:predict *store-model* newdata-store0) 0)))
