;; CMD: just TARGET=ode_ex run-guile
;; ODEs: built-in named systems, harmonic-oscillator energy check and a Lorenz phase plot.
(use-modules (oop goops) (datamunge) (ice-9 format))

(define (last-state sol) (state-at sol (- (size sol) 1)))

;; The time vector followed by the requested state components (0-based), one value per step.
(define (solution-series sol . components)
  (let* ((idx (iota (size sol)))
         (states (map (lambda (i) (state-at sol i)) idx)))
    (cons (map (lambda (i) (time-at sol i)) idx)
          (map (lambda (c) (map (lambda (s) (list-ref s c)) states)) components))))


;; Roughly C's "%g": six significant digits, no trailing ".0".
(define (fmt-g v)
  (if (zero? v)
      "0"
      (let* ((scale (expt 10 (- 5 (inexact->exact (floor (/ (log (abs v)) (log 10)))))))
             (s (number->string (/ (round (* v scale)) scale))))
        (if (string-suffix? ".0" s) (substring s 0 (- (string-length s) 2)) s))))

(define (rk4-solver step)
  (let ((options (make <ode-options>)))
    (set-method! options 'rk4)
    (set-step-size! options step)
    (make <ode-solver> options)))

;; Like the Ruby version, this example uses the built-in named systems via solve-builtin.
;; (Guile can also supply a live right-hand side by subclassing <rhs>.)

(display "=================== Every named built-in system (no director needed) ===================\n")
(define solver (make <ode-solver>))
(for-each
 (lambda (system)
   (let* ((name (car system)) (params (cadr system)) (y0 (caddr system))
          (sol (solve-builtin solver name params y0 0.0 1.0))
          (state (string-join (map fmt-g (last-state sol))
                              ", ")))
     (format #t "~22a steps=~6a final state=[~a]\n" name (steps-taken sol) state)))
 `(("exponential_decay"   (1.0)                  (1.0))
   ("logistic_growth"     (1.0 1.0)              (0.5))
   ("harmonic_oscillator" (1.0)                  (1.0 0.0))
   ("van_der_pol"         (1.0)                  (2.0 0.0))
   ("lorenz"              (10.0 28.0 ,(/ 8.0 3)) (1.0 1.0 1.0))))

(display "\n=================== Harmonic oscillator (energy conservation) ===================\n")
(define sol (solve-builtin (rk4-solver 0.01) "harmonic_oscillator" '(1.0) '(1.0 0.0) 0.0 20.0))
(define final (last-state sol))
(define x (car final))
(define v (cadr final))
(format #t "x(20) = ~,6f (cos(20) = ~,6f)\n" x (cos 20.0))
(format #t "energy x^2+v^2 = ~,6f (should stay near 1.0)\n" (+ (* x x) (* v v)))
(define series (solution-series sol 0 1))
(define ts (car series))
(define plt (r-plot-plot ts (cadr series) "l" "x(t)"))
(lines plt ts (caddr series) "v(t)")
(x-label (y-label (title plt "Harmonic Oscillator") "state") "t")
(save-svg plt "ode_harmonic_oscillator_guile.svg")
(display "wrote ode_harmonic_oscillator_guile.svg\n")

(display "\n=================== Lorenz attractor (phase plane) ===================\n")
(set! sol (solve-builtin (rk4-solver 0.005) "lorenz" (list 10.0 28.0 (/ 8.0 3)) '(1.0 1.0 1.0) 0.0 25.0))
(format #t "steps_taken = ~a\n" (steps-taken sol))
(set! series (solution-series sol 0 2))
(set! plt (r-plot-plot (cadr series) (caddr series) "l" "trajectory"))
(x-label (y-label (title plt "Lorenz Attractor (x-z phase plane)") "z") "x")
(save-svg plt "ode_lorenz_phase_plane_guile.svg")
(display "wrote ode_lorenz_phase_plane_guile.svg\n")
