(use-modules (datamunge) (ice-9 format))

;; ODESolution-state-at returns a std::vector<double> by const-reference, which the binding hands
;; back as a DVector wrapper (not a Scheme vector) -- read it with DVector-ref/DVector-length.
(define (dvec-ref v i) (DVector-ref v i))
(define (dvec->list v)
  (let loop ((i (- (DVector-length v) 1)) (acc '()))
    (if (< i 0) acc (loop (- i 1) (cons (DVector-ref v i) acc)))))

(define (last-state sol) (ODESolution-state-at sol (- (ODESolution-size sol) 1)))

;; Round to ~6 decimals for compact printing (guile format has no %g equivalent).
(define (g x) (/ (round (* x 1e6)) 1e6))

(define (state->str v)
  (string-join (map (lambda (x) (number->string (g x))) (dvec->list v)) ", "))

;; NOTE: the Python/Ruby ode examples also demonstrate a live, user-supplied RHS by subclassing
;; datamunge.RHS (a SWIG director). Stock SWIG's Guile backend generates no director code, so RHS
;; cannot be subclassed from Guile; this example uses the built-in named systems via
;; ODESolver-solve-builtin.

(format #t "=================== Every named built-in system (no director needed) ===================\n")
(define solver (new-ODESolver))
(define systems
  (list (list "exponential_decay"   (list 1.0)              (list 1.0))
        (list "logistic_growth"     (list 1.0 1.0)          (list 0.5))
        (list "harmonic_oscillator" (list 1.0)              (list 1.0 0.0))
        (list "van_der_pol"         (list 1.0)              (list 2.0 0.0))
        (list "lorenz"              (list 10.0 28.0 (/ 8.0 3.0)) (list 1.0 1.0 1.0))))
(for-each
 (lambda (s)
   (let* ((name (car s)) (params (cadr s)) (y0 (caddr s))
          (sol (ODESolver-solve-builtin solver name params y0 0.0 1.0)))
     (format #t "~22a steps=~a final state=[~a]\n"
             name (ODESolution-steps-taken-get sol) (state->str (last-state sol)))))
 systems)

(format #t "\n=================== Harmonic oscillator (energy conservation) ===================\n")
(define options (new-ODEOptions))
(ODEOptions-method-set options (StepMethod-RK4))
(ODEOptions-step-size-set options 0.01)
(set! solver (new-ODESolver options))
(define sol (ODESolver-solve-builtin solver "harmonic_oscillator" (list 1.0) (list 1.0 0.0) 0.0 20.0))
(define final (last-state sol))
(define xh (dvec-ref final 0))
(define vh (dvec-ref final 1))
(format #t "x(20) = ~,6f (cos(20) = ~,6f)\n" xh (cos 20))
(format #t "energy x^2+v^2 = ~,6f (should stay near 1.0)\n" (+ (* xh xh) (* vh vh)))

(define t-series '())
(define x-series '())
(define v-series '())
(do ((i 0 (+ i 1))) ((= i (ODESolution-size sol)))
  (set! t-series (cons (ODESolution-time-at sol i) t-series))
  (let ((st (ODESolution-state-at sol i)))
    (set! x-series (cons (dvec-ref st 0) x-series))
    (set! v-series (cons (dvec-ref st 1) v-series))))
(set! t-series (reverse t-series))
(set! x-series (reverse x-series))
(set! v-series (reverse v-series))
(define plot (RPlot-plot t-series x-series "l" "x(t)"))
(RPlot-lines plot t-series v-series "v(t)")
(Plot-title plot "Harmonic Oscillator")
(Plot-x-label plot "t")
(Plot-y-label plot "state")
(Plot-save-svg plot "ode_harmonic_oscillator_guile.svg")
(format #t "wrote ode_harmonic_oscillator_guile.svg\n")

(format #t "\n=================== Lorenz attractor (phase plane) ===================\n")
(set! options (new-ODEOptions))
(ODEOptions-method-set options (StepMethod-RK4))
(ODEOptions-step-size-set options 0.005)
(set! solver (new-ODESolver options))
(set! sol (ODESolver-solve-builtin solver "lorenz" (list 10.0 28.0 (/ 8.0 3.0)) (list 1.0 1.0 1.0) 0.0 25.0))
(format #t "steps_taken = ~a\n" (ODESolution-steps-taken-get sol))

(define lx '())
(define lz '())
(do ((i 0 (+ i 1))) ((= i (ODESolution-size sol)))
  (let ((st (ODESolution-state-at sol i)))
    (set! lx (cons (dvec-ref st 0) lx))
    (set! lz (cons (dvec-ref st 2) lz))))
(set! lx (reverse lx))
(set! lz (reverse lz))
(define phase (RPlot-plot lx lz "l" "trajectory"))
(Plot-title phase "Lorenz Attractor (x-z phase plane)")
(Plot-x-label phase "x")
(Plot-y-label phase "z")
(Plot-save-svg phase "ode_lorenz_phase_plane_guile.svg")
(format #t "wrote ode_lorenz_phase_plane_guile.svg\n")
