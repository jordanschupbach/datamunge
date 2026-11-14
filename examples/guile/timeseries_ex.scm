;; CMD: just TARGET=timeseries_ex run-guile
;; Time series: ARIMA, seasonal ARIMA and exponential smoothing (SES, Holt, Holt-Winters).
(use-modules (oop goops) (datamunge) (ice-9 format))

(define pi (* 4 (atan 1)))
(define rng (seed->random-state 42))
(define (seed! n) (set! rng (seed->random-state n)))
(define (uniform) (random:uniform rng))
;; Box-Muller on (0, 1] uniforms (avoids log(0)); seed! reseeds the generator.
(define (gauss)
  (* (sqrt (* -2 (log (- 1 (uniform))))) (cos (* 2 pi (- 1 (uniform))))))

(define (join xs) (string-join (map number->string xs) ", "))

(display "=================== ARIMA(1,1,1) on a simulated random walk with drift ===================\n")
(seed! 42)
(define y
  (let loop ((i 0) (level 100.0) (prev-shock 0.0) (acc '()))
    (if (= i 150)
        (reverse acc)
        (let* ((shock (gauss))
               ;; integrated ARMA(0,1) increments with drift
               (level (+ level 0.3 (* 0.4 prev-shock) shock)))
          (loop (+ i 1) level shock (cons level acc))))))

(define options (make <arima-options>))
(set-p! options 1)
(set-d! options 1)
(set-q! options 1)
(set-de-population-size! options 80)
(set-de-max-generations! options 400)
(define model (make <arima> y options))

(format #t "AR coefficient: ~a\n" (car (ar-coefficients model)))
(format #t "MA coefficient: ~a\n" (car (ma-coefficients model)))
(format #t "sigma^2: ~a, AIC: ~a, BIC: ~a\n" (sigma2 model) (aic model) (bic model))

(let ((fc (forecast-with-intervals model 6)))
  (format #t "6-step forecast: ~a\n" (join (car fc)))
  (format #t "forecast std. errors: ~a\n" (join (cdr fc))))

(display "\n=================== SARIMA(1,0,0)(1,1,0)_12 on a seasonal series ===================\n")
(seed! 7)
(define s
  (let loop ((i 0) (prev 0.0) (acc '()))
    (if (= i 120)
        (reverse acc)
        (let ((prev (+ (* 0.5 prev) (gauss))))
          (loop (+ i 1) prev
                (cons (+ 20 (* 0.2 i) (* 5 (sin (/ (* 2 pi i) 12))) prev) acc))))))

(define options2 (make <arima-options>))
(set-p! options2 1)
(set-seasonal-p! options2 1)
(set-seasonal-d! options2 1)
(set-seasonal-period! options2 12)
(set-de-population-size! options2 100)
(set-de-max-generations! options2 500)
(define model2 (make <arima> s options2))
(format #t "AR coefficient: ~a, seasonal AR coefficient: ~a\n"
        (car (ar-coefficients model2)) (car (seasonal-ar-coefficients model2)))
(format #t "12-step forecast: ~a\n" (join (forecast model2 12)))

(display "\n=================== Exponential smoothing: Holt-Winters on retail-style seasonal data ===================\n")
(define seasonal-shape '(0.8 0.75 0.9 0.95 1.0 1.05 1.1 1.05 1.0 1.1 1.3 1.6))
(seed! 11)
(define y2
  (map (lambda (i)
         (+ (* (+ 100 (* 2 i)) (list-ref seasonal-shape (modulo i 12))) (* (gauss) 3)))
       (iota 48)))

(define es-options (make <exponential-smoothing-options>))
(set-trend! es-options 'additive)
(set-seasonal! es-options 'multiplicative)
(set-seasonal-period! es-options 12)
(set-de-population-size! es-options 60)
(set-de-max-generations! es-options 300)
(define es-model (make <exponential-smoothing> y2 es-options))
(format #t "alpha=~a beta=~a gamma=~a\n" (alpha es-model) (beta es-model) (gamma es-model))
(format #t "sigma^2: ~a, AIC: ~a\n" (sigma2 es-model) (aic es-model))
(format #t "12-month forecast: ~a\n" (join (forecast es-model 12)))

(display "\n=================== Simple exponential smoothing vs. Holt's linear trend ===================\n")
(define flat '(50.2 49.8 50.5 49.6 50.1 50.3 49.9 50.0 50.4 49.7))

(define ses (make <exponential-smoothing> flat (make <exponential-smoothing-options>)))
(format #t "SES alpha= ~a\n" (alpha ses))
(format #t "SES 5-step forecast: ~a\n" (join (forecast ses 5)))

(define holt-options (make <exponential-smoothing-options>))
(set-trend! holt-options 'additive)
(define holt (make <exponential-smoothing> flat holt-options))
(format #t "Holt alpha=~a beta=~a\n" (alpha holt) (beta holt))
(format #t "Holt 5-step forecast: ~a\n" (join (forecast holt 5)))
