;; CMD: just TARGET=optim_advanced_ex run-guile
;; Advanced datamunge optimizers (ProximalGradient, FISTA, LevenbergMarquardt, Newton, TrustRegionNewton, AugmentedLagrangian, SQP, InteriorPoint, BayesianOptimization, RBFGaussianProcessSurrogate) from Guile.
(use-modules (oop goops) (datamunge) (ice-9 format))

;; Every one of these optimizers takes an interface (<proximal-function> / <hessian-function> /
;; <equality-constrained-function> / <inequality-constrained-function> / <residual-function>, or
;; <arbitrary-function> for <bayesian-optimization>) that the C++ library leaves abstract: callers
;; implement the objective. From Guile, subclass the interface class with define-class and
;; specialize its methods with define-method; the optimizer then calls the Scheme methods.
;; Pass the coordinates as a vector: it is updated in place.

;; f(x, y) = (x - 1)^2 + 10 (y + 2)^2 with exact gradient and Hessian.
(define-class <quadratic> (<hessian-function>))
(define-method (evaluate (f <quadratic>) c)
  (+ (expt (- (list-ref c 0) 1) 2) (* 10 (expt (+ (list-ref c 1) 2) 2))))
(define-method (gradient (f <quadratic>) c)
  (list (* 2 (- (list-ref c 0) 1)) (* 20 (+ (list-ref c 1) 2))))
(define-method (hessian (f <quadratic>) c)
  '((2.0 0.0) (0.0 20.0)))

;; Residuals of y = a * exp(b t) against exact data from a = 2, b = 0.5.
(define-class <exponential> (<residual-function>)
  (ts #:init-keyword #:ts))
(define-method (residuals (f <exponential>) p)
  (map (lambda (t) (- (* (car p) (exp (* (cadr p) t))) (* 2 (exp (* 0.5 t)))))
       (slot-ref f 'ts)))
(define-method (jacobian (f <exponential>) p)
  (map (lambda (t) (list (exp (* (cadr p) t)) (* (car p) t (exp (* (cadr p) t)))))
       (slot-ref f 'ts)))

;; Leaves gradient and hessian out.
(define-class <incomplete> (<hessian-function>))
(define-method (evaluate (f <incomplete>) c) 0.0)

(display "=================== RBFGaussianProcessSurrogate (concrete BayesianSurrogate) ===================\n")
;; Fit a 1D surrogate to a simple bowl y = (x-1)^2 sampled at a few points,
;; then check the acquisition function favors an unexplored region.
(define xs '(0.0 0.5 1.0 1.5 2.0 3.0))
(define ys (map (lambda (x) (expt (- x 1) 2)) xs))
(define surrogate (make <rbf-gaussian-process-surrogate> 1.0 1e-6))
(fit surrogate (map list xs) ys)
(define incumbent (min 0.0 (apply min ys))) ; best observed value so far
(format #t "incumbent (best observed y): ~a\n" incumbent)
(for-each (lambda (x)
            (format #t "acquisition(x=~,2f) = ~,6f\n" x (acquisition surrogate (list x) incumbent)))
          '(0.25 1.0 2.5))

(display "\n=================== Options structs + optimizer construction (all 10 new classes) ===================\n")
;; Every new Options struct and optimizer class constructs cleanly, proving
;; the binding for the whole batch is wired correctly.
(define pg (make <proximal-gradient>
             (let ((o (make <proximal-gradient-options>))) (set-step-size! o 0.05) o)))
(define fista (make <fista> (make <fista-options>)))
(define lm (make <levenberg-marquardt>
             (let ((o (make <levenberg-marquardt-options>))) (set-max-iterations! o 200) o)))
(define newton (make <newton> (make <newton-options>)))
(define trn (make <trust-region-newton> (make <trust-region-newton-options>)))
(define al (make <augmented-lagrangian> (make <augmented-lagrangian-options>)))
(define sqp (make <sqp> (make <sqp-options>)))
(define ip (make <interior-point> (make <interior-point-options>)))
(define bo (make <bayesian-optimization>
             (let ((o (make <bayesian-optimization-options>))) (set-initial-samples! o 8) o)))
(for-each (lambda (optimizer)
            (let ((name (string-trim-both (symbol->string (class-name (class-of optimizer)))
                                           (char-set #\< #\>))))
              (format #t "~a: ~a()\n" name name)))
          (list pg fista lm newton trn al sqp ip bo))

(display "\n=================== Guile objectives (subclassing the C++ interfaces) ===================\n")
;; Newton's method on the quadratic bowl.
(let* ((start (vector 5.0 5.0))
       (value (optimize newton (make <quadratic>) start)))
  (format #t "Newton: f = ~,6g at (~,6f, ~,6f) (exact minimum 0 at (1, -2))\n"
          value (vector-ref start 0) (vector-ref start 1)))

;; Levenberg-Marquardt fitting y = a * exp(b t).
(let* ((params (vector 1.0 0.1))
       (sse (optimize lm (make <exponential> #:ts '(0.0 0.5 1.0 1.5 2.0)) params)))
  (format #t "LevenbergMarquardt: residual = ~,3g, a = ~,6f, b = ~,6f (exact: 2, 0.5)\n"
          sse (vector-ref params 0) (vector-ref params 1)))

;; An interface method a subclass leaves out raises when the optimizer calls it.
(define err
  (catch #t
    (lambda () (optimize newton (make <incomplete>) (vector 0.0 0.0)) #f)
    (lambda (key . args) (format #f "~a ~s" key args))))
(format #t "missing gradient(): ok=~a, error mentions it: ~a\n"
        (if err "false" "true")
        (if (and err (string-contains err "gradient")) "true" "false"))
