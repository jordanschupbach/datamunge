;; CMD: just TARGET=lmm_ex run-guile
;; Linear mixed models: random intercept (penguins) and random intercept + slope (simulated schools).
(use-modules (oop goops) (datamunge) ((datamunge data) #:select (penguins)))

(define rng (seed->random-state 2024))
(define (uniform) (random:uniform rng))
;; Box-Muller on (0, 1] uniforms (avoids log(0)).
(define pi (* 4 (atan 1.0)))
(define (gauss)
  (* (sqrt (* -2 (log (- 1 (uniform))))) (cos (* 2 pi (- 1 (uniform))))))

(display "=================== Random intercept on a real dataset (penguins) ===================\n")
(print-summary (make <lmm> (penguins) "body_mass_g ~ flipper_length_mm + bill_length_mm + (1 | species)"))

(display "\n=================== Random intercept + slope on a simulated multi-school dataset ===================\n")
;; 30 schools, ~25 students each: score depends on study_hours, but both the baseline score
;; and the return on an extra hour of study vary by school -- the textbook case for a
;; random-intercept-and-slope model.
(define n-schools 30)
;; A list of N values drawn in order from THUNK.
(define (draw n thunk)
  (let loop ((i 0) (acc '()))
    (if (= i n) (reverse acc) (loop (+ i 1) (cons (thunk) acc)))))
(define school-intercept (draw n-schools (lambda () (* (gauss) 6))))
(define school-slope (draw n-schools (lambda () (* (gauss) 1.2))))
(define true-intercept 60.0)
(define true-slope 3.0)

(define school '())
(define study-hours '())
(define score '())
(do ((s 0 (+ s 1))) ((= s n-schools))
  (do ((j (+ 15 (random 21 rng)) (- j 1))) ((zero? j))
    (let* ((hours (* (uniform) 10))
           (y (+ true-intercept (list-ref school-intercept s)
                 (* (+ true-slope (list-ref school-slope s)) hours) (* (gauss) 4))))
      (set! school (cons (exact->inexact s) school))
      (set! study-hours (cons hours study-hours))
      (set! score (cons y score)))))

(define df (make <data-frame>))
(add-numeric-column df "school" (reverse school))
(add-numeric-column df "study_hours" (reverse study-hours))
(add-numeric-column df "score" (reverse score))

(define model (make <lmm> df "score ~ study_hours + (1 + study_hours | school)"))
(print-summary model)

(format #t "\nTrue generating values: intercept=~a, slope=~a, random-intercept SD=6.0, random-slope SD=1.2, residual SD=4.0\n"
        true-intercept true-slope)

(display "\n--- Best Linear Unbiased Predictors (BLUPs) for a few schools ---\n")
(define group-names (group-labels model))
(for-each
 (lambda (idx)
   (let ((shifts (random-effects-for-group model idx)))
     (format #t "school ~a: intercept shift=~a, slope shift=~a\n"
             (list-ref group-names idx) (car shifts) (cadr shifts))))
 (iota 3))

(display "\n--- Prediction: population-level vs. school-adjusted ---\n")
(define population (make <data-frame>))
(define school0 (make <data-frame>))
(add-numeric-column population "study_hours" '(5.0))
(add-numeric-column school0 "study_hours" '(5.0))
(add-numeric-column school0 "school" '(0.0))
(format #t "5 study hours, unseen school:      ~a (fixed effects only)\n" (car (predict model population)))
(format #t "5 study hours, school 0 (known):    ~a (fixed effects + school 0's BLUP)\n" (car (predict model school0)))
