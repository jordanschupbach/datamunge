;; CMD: just TARGET=glmm_ex run-guile
;; Generalized linear mixed models: a binomial GLMM on penguins and a Poisson GLMM on simulated store data.
(use-modules (oop goops) (datamunge) ((datamunge data) #:select (penguins))
             ((srfi srfi-1) #:select (any)) (ice-9 format))

(define pi (* 4 (atan 1)))
(define rng (seed->random-state 4242))
(define (uniform) (random:uniform rng))
(define (gauss) ; 1 - u keeps log() away from 0
  (* (sqrt (* -2 (log (- 1 (uniform))))) (cos (* 2 pi (uniform)))))

(display "=================== Binomial (logistic) mixed model on a real dataset (penguins) ===================\n")
;; Predicting sex from body mass, grouped by island: unlike species (which is almost perfectly
;; confounded with island in this dataset -- Gentoo penguins occur on Biscoe island only), sex
;; is not tied to location, so this is a well-behaved fit.
(define pg (penguins))
(define rows
  (filter (lambda (i)
            (not (any (lambda (c) (null? pg c i)) '("sex" "body_mass_g" "island"))))
          (iota (nrows pg))))
(define sex-df (make <data-frame>))
(add-numeric-column sex-df "is_male"
                    (map (lambda (i) (if (string=? (string-at pg "sex" i) "male") 1.0 0.0)) rows))
(add-numeric-column sex-df "body_mass_g" (map (lambda (i) (numeric-at pg "body_mass_g" i)) rows))
(add-string-column sex-df "island" (map (lambda (i) (string-at pg "island" i)) rows))
(print-summary (make <glmm> sex-df "is_male ~ body_mass_g + (1 | island)" "binomial"))

(display "\n=================== Poisson mixed model on simulated multi-site count data ===================\n")
;; 25 stores, ~20 days each: daily visit counts depend on a promo intensity score, but both
;; the baseline traffic and the promo's effectiveness vary by store.
(define n-stores 25)
(define store-effect (map (lambda (_) (* (gauss) 0.4)) (iota n-stores)))
(define true-intercept 2.0)
(define true-slope 0.3)

;; Knuth's Poisson sampler.
(define (poisson lam)
  (let ((threshold (exp (- lam))))
    (let loop ((k 1) (p (uniform)))
      (if (> p threshold)
          (loop (+ k 1) (* p (uniform)))
          (- k 1)))))

(define store '())
(define promo '())
(define visits '())
(do ((s 0 (+ s 1))) ((= s n-stores))
  (do ((day 0 (+ day 1)) (n-days (+ 15 (random 11 rng)))) ((= day n-days))
    (let* ((promo-intensity (* (uniform) 3))
           (lam (exp (+ true-intercept (list-ref store-effect s) (* true-slope promo-intensity)))))
      (set! store (cons (exact->inexact s) store))
      (set! promo (cons promo-intensity promo))
      (set! visits (cons (exact->inexact (poisson lam)) visits)))))
(define df (make <data-frame>))
(add-numeric-column df "store" (reverse store))
(add-numeric-column df "promo" (reverse promo))
(add-numeric-column df "visits" (reverse visits))
(define store-model (make <glmm> df "visits ~ promo + (1 | store)" "poisson"))

(print-summary store-model)

(format #t "\nTrue generating values: intercept=~a, slope=~a, random-intercept SD (log scale)=0.4\n"
        true-intercept true-slope)

(display "\n--- BLUPs for a few stores ---\n")
(define labels (group-labels store-model))
(for-each (lambda (idx)
            (format #t "store ~a: intercept shift=~a\n"
                    (list-ref labels idx) (car (random-effects-for-group store-model idx))))
          (iota 3))

(display "\n--- Prediction: population-level vs. store-adjusted ---\n")
(define population (make <data-frame>))
(define store0 (make <data-frame>))
(add-numeric-column population "promo" '(1.5))
(add-numeric-column store0 "promo" '(1.5))
(add-numeric-column store0 "store" '(0.0))
(format #t "promo=1.5, unseen store:   ~a expected visits (fixed effects only)\n"
        (car (predict store-model population)))
(format #t "promo=1.5, store 0 (known): ~a expected visits (fixed effects + store 0's BLUP)\n"
        (car (predict store-model store0)))
