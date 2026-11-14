;; CMD: just TARGET=autodiff_ex run-guile
;; Automatic differentiation: forward mode (Dual), reverse mode (Tape/Var), Hessians (HyperDual).
(use-modules (oop goops) (datamunge))

;; Note: the C++ autodiff_ex.cpp additionally demonstrates the generic template drivers
;; (derivative/gradient_forward/gradient_reverse/jacobian_forward/hessian), which can't be
;; bound. This port demonstrates the same ideas with the bound named-method API.
;; (index var) is the Var's position in the adjoint list (backward tape out) returns.

;; The Rosenbrock function built from named operations; `one` is the constant 1 in the
;; number type at hand (Dual, Var or HyperDual).
(define (rosenbrock one x0 x1)
  (let ((a (subtract one x0))
        (b (subtract x1 (multiply x0 x0))))
    (add (multiply a a) (multiply-scalar (multiply b b) 100.0))))

(define (rosenbrock-dual x0 x1) (rosenbrock (make <dual> 1.0 0.0) x0 x1))
(define (rosenbrock-var tape x0 x1) (rosenbrock (make <var> tape 1.0) x0 x1))
(define (rosenbrock-hyper-dual x0 x1) (rosenbrock (make <hyper-dual> 1.0 0.0 0.0 0.0) x0 x1))

(define (rosenbrock-value x0 x1)
  (+ (expt (- 1.0 x0) 2) (* 100.0 (expt (- x1 (* x0 x0)) 2))))

(define (adjoint adjoints v) (list-ref adjoints (index v)))

(display "=================== Forward mode: scalar derivative ===================\n")
(let* ((x0 (make <dual> 2.0 1.0)) ; seed derivative = 1 to read df/dx directly
       (f (subtract (multiply (multiply x0 x0) x0) (multiply-scalar x0 2.0)))) ; x^3 - 2x
  (format #t "f(x) = x^3 - 2x, f'(2) = ~a (exact: 10)\n" (derivative f)))

(display "\n=================== Reverse mode: build a graph by hand ===================\n")
(let* ((tape (make <tape>))
       (a (make <var> tape 2.0))
       (b (make <var> tape 3.0))
       (y (add (multiply a b) (sin a))))
  (format #t "y = a*b + sin(a) at a=2, b=3 -> y = ~a\n" (value y))
  (let ((adj (backward tape y)))
    (format #t "dy/da = ~a (exact: b + cos(a))\n" (adjoint adj a))
    (format #t "dy/db = ~a (exact: a)\n" (adjoint adj b))))

(display "\n=================== Forward vs reverse mode agree on the Rosenbrock function ===================\n")
(let* ((px 0.0) (py 0.0)
       ;; Forward mode: one Dual pass per partial derivative, seeding the direction of interest.
       (gx (derivative (rosenbrock-dual (make <dual> px 1.0) (make <dual> py 0.0))))
       (gy (derivative (rosenbrock-dual (make <dual> px 0.0) (make <dual> py 1.0))))
       ;; Reverse mode: one Tape/Var pass computes every partial at once.
       (tape (make <tape>))
       (vx (make <var> tape px))
       (vy (make <var> tape py))
       (grad (backward tape (rosenbrock-var tape vx vy))))
  (format #t "f(0,0) = ~a\n" (rosenbrock-value px py))
  (format #t "gradient (forward mode): [~a, ~a]\n" gx gy)
  (format #t "gradient (reverse mode): [~a, ~a]\n" (adjoint grad vx) (adjoint grad vy)))

(display "\n=================== Jacobian of a vector-valued function ===================\n")
;; f(x,y) = [x^2, xy, y^3] at (2,3) -- one Dual pass per (output, input) pair.
(define (vector-fn x y)
  (list (multiply x x) (multiply x y) (multiply (multiply y y) y)))
(let ((columns (map (lambda (seed)
                      (map derivative (vector-fn (make <dual> 2.0 (car seed))
                                                 (make <dual> 3.0 (cdr seed)))))
                    '((1.0 . 0.0) (0.0 . 1.0)))))
  (display "f(x,y) = [x^2, xy, y^3] at (2,3), Jacobian:\n")
  (for-each (lambda (d/dx d/dy) (format #t "  [~a, ~a]\n" d/dx d/dy))
            (car columns) (cadr columns)))

(display "\n=================== Hessian via second-order forward mode (HyperDual) ===================\n")
;; One HyperDual pass per (i, j) pair reads off d^2f/dxi dxj directly from eps1eps2.
(let* ((mx 1.0) (my 1.0) ; the Rosenbrock function's minimum
       (seeds '((1.0 0.0) (0.0 1.0)))
       (h (map (lambda (e1)
                 (map (lambda (e2)
                        (eps1eps2 (rosenbrock-hyper-dual
                                   (make <hyper-dual> mx (car e1) (car e2) 0.0)
                                   (make <hyper-dual> my (cadr e1) (cadr e2) 0.0))))
                      seeds))
               seeds)))
  (display "Hessian of the Rosenbrock function at its minimum (1,1):\n")
  (for-each (lambda (r) (format #t "  [~a, ~a]\n" (car r) (cadr r))) h))

(display "\n=================== Gradient descent driven by reverse-mode gradients ===================\n")
(let ((px -1.2) (py 1.0) ; the classic Rosenbrock start
      (learning-rate 0.001)
      (n-steps 2000))
  (do ((step 0 (+ step 1))) ((= step n-steps))
    (let* ((tape (make <tape>))
           (vx (make <var> tape px))
           (vy (make <var> tape py))
           (vf (rosenbrock-var tape vx vy))
           (grad (backward tape vf))
           (loss (value vf)))
      (set! px (- px (* learning-rate (adjoint grad vx))))
      (set! py (- py (* learning-rate (adjoint grad vy))))
      (when (or (= step 0) (= step (- n-steps 1)))
        (format #t "step ~a: loss = ~a, x = [~a, ~a]\n" step loss px py))))
  (display "(true minimum is at [1, 1] with loss 0)\n"))
