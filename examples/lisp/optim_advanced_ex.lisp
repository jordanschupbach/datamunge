;;;; Advanced datamunge optimizers (ProximalGradient, FISTA, LevenbergMarquardt, Newton, TrustRegionNewton,
;;;; AugmentedLagrangian, SQP, InteriorPoint, BayesianOptimization, RBFGaussianProcessSurrogate) from Lisp.
;;;;
;;;; Every one of these optimizers takes an interface (ProximalFunction / HessianFunction /
;;;; EqualityConstrainedFunction / InequalityConstrainedFunction / ResidualFunction, or
;;;; ArbitraryFunction for BayesianOptimization) that the C++ library leaves abstract: callers
;;;; implement the objective. From Lisp, subclass the interface class with DEFCLASS and define its
;;;; methods with DM:DEFINE-OVERRIDE; the optimizer then calls the Lisp methods. Coordinates are
;;;; updated in place through a DM:REF cell (the C++ parameter is a non-const std::vector<double>&).
(require "asdf")
(asdf:load-system "datamunge")

(setf *read-default-float-format* 'double-float)

(format t "=================== RBFGaussianProcessSurrogate (concrete BayesianSurrogate) ===================~%")
;; Fit a 1D surrogate to a simple bowl y = (x-1)^2 sampled at a few points,
;; then check the acquisition function favors an unexplored region.
(let* ((xs '(0d0 0.5d0 1d0 1.5d0 2d0 3d0))
       (ys (mapcar (lambda (x) (expt (- x 1d0) 2)) xs))
       (surrogate (dm:make-rbf-gaussian-process-surrogate 1d0 1d-6))
       (incumbent 0d0))                 ; best observed value so far (min of ys)
  (dm:fit surrogate (mapcar #'vector xs) ys)
  (dolist (y ys) (when (< y incumbent) (setf incumbent y)))
  (format t "incumbent (best observed y): ~a~%" incumbent)
  (dolist (x '(0.25d0 1d0 2.5d0))
    (format t "acquisition(x=~,2f) = ~,6f~%" x (dm:acquisition surrogate (vector x) incumbent))))

(format t "~%=================== Options structs + optimizer construction (all 10 new classes) ===================~%")
;; Every new Options struct and optimizer class constructs cleanly, proving
;; the binding for the whole batch is wired correctly.
(defparameter *pg*
  (let ((options (dm:make-proximal-gradient-options)))
    (setf (dm:step-size options) 0.05d0)
    (dm:make-proximal-gradient options)))
(format t "ProximalGradient: ~a~%" *pg*)

(defparameter *fista* (dm:make-fista (dm:make-fista-options)))
(format t "FISTA: ~a~%" *fista*)

(defparameter *lm*
  (let ((options (dm:make-levenberg-marquardt-options)))
    (setf (dm:max-iterations options) 200)
    (dm:make-levenberg-marquardt options)))
(format t "LevenbergMarquardt: ~a~%" *lm*)

(defparameter *newton* (dm:make-newton (dm:make-newton-options)))
(format t "Newton: ~a~%" *newton*)

(defparameter *trn* (dm:make-trust-region-newton (dm:make-trust-region-newton-options)))
(format t "TrustRegionNewton: ~a~%" *trn*)

(defparameter *al* (dm:make-augmented-lagrangian (dm:make-augmented-lagrangian-options)))
(format t "AugmentedLagrangian: ~a~%" *al*)

(defparameter *sqp* (dm:make-sqp (dm:make-sqp-options)))
(format t "SQP: ~a~%" *sqp*)

(defparameter *ip* (dm:make-interior-point (dm:make-interior-point-options)))
(format t "InteriorPoint: ~a~%" *ip*)

(defparameter *bo*
  (let ((options (dm:make-bayesian-optimization-options)))
    (setf (dm:initial-samples options) 8)
    (dm:make-bayesian-optimization options)))
(format t "BayesianOptimization: ~a~%" *bo*)

(format t "~%=================== Lisp objectives (subclassing the C++ interfaces) ===================~%")
;; Newton's method on f(x, y) = (x - 1)^2 + 10 (y + 2)^2 with exact gradient and Hessian.
(defclass quadratic (dm:hessian-function) ())
(dm:define-override dm:evaluate ((self quadratic) c)
  (+ (expt (- (elt c 0) 1) 2) (* 10 (expt (+ (elt c 1) 2) 2))))
(dm:define-override dm:gradient ((self quadratic) c)
  (vector (* 2 (- (elt c 0) 1)) (* 20 (+ (elt c 1) 2))))
(dm:define-override dm:hessian ((self quadratic) c)
  (declare (ignore c))
  #(#(2d0 0d0) #(0d0 20d0)))

(let* ((start (dm:ref #(5d0 5d0)))
       (value (dm:optimize *newton* (make-instance 'quadratic) start))
       (x (dm:ref-value start)))
  (format t "Newton: f = ~,6g at (~,6f, ~,6f) (exact minimum 0 at (1, -2))~%" value (elt x 0) (elt x 1)))

;; Levenberg-Marquardt fitting y = a * exp(b t) to exact data from a = 2, b = 0.5.
(defparameter *ts* '(0d0 0.5d0 1d0 1.5d0 2d0))
(defclass exponential (dm:residual-function) ())
(dm:define-override dm:residuals ((self exponential) p)
  (map 'vector (lambda (tt) (- (* (elt p 0) (exp (* (elt p 1) tt))) (* 2 (exp (* 0.5d0 tt))))) *ts*))
(dm:define-override dm:jacobian ((self exponential) p)
  (map 'vector (lambda (tt) (vector (exp (* (elt p 1) tt)) (* (elt p 0) tt (exp (* (elt p 1) tt))))) *ts*))

(let* ((params (dm:ref #(1d0 0.1d0)))
       (sse (dm:optimize *lm* (make-instance 'exponential) params))
       (p (dm:ref-value params)))
  (format t "LevenbergMarquardt: residual = ~,3g, a = ~,6f, b = ~,6f (exact: 2, 0.5)~%" sse (elt p 0) (elt p 1)))

;; An interface method a subclass leaves out signals an error when the optimizer calls it.
(defclass incomplete (dm:hessian-function) ())
(dm:define-override dm:evaluate ((self incomplete) c)
  (declare (ignore c))
  0d0)
(multiple-value-bind (result err)
    (ignore-errors (dm:optimize *newton* (make-instance 'incomplete) (dm:ref #(0d0 0d0))))
  (declare (ignore result))
  (format t "missing gradient(): ok=~a, error mentions it: ~a~%"
          (if err "false" "true")
          (if (and err (search "gradient" (princ-to-string err))) "true" "false")))
