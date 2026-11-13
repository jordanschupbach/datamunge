;; CMD: just TARGET=autodiff_ex run-lisp

;;;; Automatic differentiation: forward mode (Dual), reverse mode (Tape/Var), Hessians (HyperDual).
(require "asdf")
(asdf:load-system "datamunge")
(setf *read-default-float-format* (quote double-float)) ; print doubles without a d0 suffix

;;; Note: the C++ autodiff_ex.cpp additionally demonstrates the generic template drivers
;;; (derivative/gradient_forward/gradient_reverse/jacobian_forward/hessian), which can't be
;;; bound. This port demonstrates the same ideas with the bound named-method API.

(defun rosenbrock-dual (x0 x1)
  (let ((a (dm:subtract (dm:make-dual 1d0 0d0) x0))
        (b (dm:subtract x1 (dm:multiply x0 x0))))
    (dm:add (dm:multiply a a) (dm:multiply-scalar (dm:multiply b b) 100d0))))

(defun rosenbrock-value (x0 x1)
  (let ((a (- 1d0 x0))
        (b (- x1 (* x0 x0))))
    (+ (* a a) (* 100d0 b b))))

(defun rosenbrock-var (tape x0 x1)
  "Build the Rosenbrock graph on TAPE; returns the output Var."
  (let ((a (dm:subtract (dm:make-var tape 1d0) x0))
        (b (dm:subtract x1 (dm:multiply x0 x0))))
    (dm:add (dm:multiply a a) (dm:multiply-scalar (dm:multiply b b) 100d0))))

(format t "=================== Forward mode: scalar derivative ===================~%")
(let* ((x0 (dm:make-dual 2d0 1d0))  ; seed derivative = 1 to read df/dx directly
       (f (dm:subtract (dm:multiply (dm:multiply x0 x0) x0)
                       (dm:multiply-scalar x0 2d0))))  ; x^3 - 2x
  (format t "f(x) = x^3 - 2x, f'(2) = ~a (exact: 10)~%" (dm:derivative f)))

(format t "~%=================== Reverse mode: build a graph by hand ===================~%")
(let* ((tape (dm:make-tape))
       (a (dm:make-var tape 2d0))
       (b (dm:make-var tape 3d0))
       (y (dm:add (dm:multiply a b) (dm:sin a))))
  (format t "y = a*b + sin(a) at a=2, b=3 -> y = ~a~%" (dm:value y))
  (let ((adjoint (dm:backward tape y)))
    (format t "dy/da = ~a (exact: b + cos(a))~%" (aref adjoint (dm:index a)))
    (format t "dy/db = ~a (exact: a)~%" (aref adjoint (dm:index b)))))

(format t "~%=================== Forward vs reverse mode agree on the Rosenbrock function ===================~%")
(let* ((px 0d0) (py 0d0)
       ;; Forward mode: one Dual pass per partial derivative, seeding the direction of interest.
       (gx (dm:derivative (rosenbrock-dual (dm:make-dual px 1d0) (dm:make-dual py 0d0))))
       (gy (dm:derivative (rosenbrock-dual (dm:make-dual px 0d0) (dm:make-dual py 1d0))))
       ;; Reverse mode: one Tape/Var pass computes every partial at once.
       (tape2 (dm:make-tape))
       (vx (dm:make-var tape2 px))
       (vy (dm:make-var tape2 py))
       (grad-rev (dm:backward tape2 (rosenbrock-var tape2 vx vy))))
  (format t "f(0,0) = ~a~%" (rosenbrock-value px py))
  (format t "gradient (forward mode): [~a, ~a]~%" gx gy)
  (format t "gradient (reverse mode): [~a, ~a]~%"
          (aref grad-rev (dm:index vx)) (aref grad-rev (dm:index vy))))

(format t "~%=================== Jacobian of a vector-valued function ===================~%")
;; f(x,y) = [x^2, xy, y^3] at (2,3) -- one Dual pass per (output, input) pair.
(defun vector-fn (x y)
  (list (dm:multiply x x) (dm:multiply x y) (dm:multiply (dm:multiply y y) y)))

(let ((jac (make-array '(3 2) :element-type 'double-float :initial-element 0d0)))
  (loop for (col seed-x seed-y) in '((0 1d0 0d0) (1 0d0 1d0))
        do (loop for output in (vector-fn (dm:make-dual 2d0 seed-x) (dm:make-dual 3d0 seed-y))
                 for row from 0
                 do (setf (aref jac row col) (dm:derivative output))))
  (format t "f(x,y) = [x^2, xy, y^3] at (2,3), Jacobian:~%")
  (dotimes (row 3)
    (format t "  [~a, ~a]~%" (aref jac row 0) (aref jac row 1))))

(format t "~%=================== Hessian via second-order forward mode (HyperDual) ===================~%")
;; One HyperDual pass per (i, j) pair reads off d^2f/dxi dxj directly from eps1eps2.
(defun rosenbrock-hyperdual (x0 x1)
  (let ((a (dm:subtract (dm:make-hyper-dual 1d0 0d0 0d0 0d0) x0))
        (b (dm:subtract x1 (dm:multiply x0 x0))))
    (dm:add (dm:multiply a a) (dm:multiply-scalar (dm:multiply b b) 100d0))))

(let ((mx 1d0) (my 1d0)  ; the Rosenbrock function's minimum
      (seeds #((1d0 0d0) (0d0 1d0)))
      (h (make-array '(2 2) :element-type 'double-float :initial-element 0d0)))
  (dotimes (i 2)
    (dotimes (j 2)
      (destructuring-bind (e1x e1y) (aref seeds i)
        (destructuring-bind (e2x e2y) (aref seeds j)
          (setf (aref h i j)
                (dm:eps1eps2 (rosenbrock-hyperdual (dm:make-hyper-dual mx e1x e2x 0d0)
                                                   (dm:make-hyper-dual my e1y e2y 0d0))))))))
  (format t "Hessian of the Rosenbrock function at its minimum (1,1):~%")
  (dotimes (i 2)
    (format t "  [~a, ~a]~%" (aref h i 0) (aref h i 1))))

(format t "~%=================== Gradient descent driven by reverse-mode gradients ===================~%")
(let ((point (vector -1.2d0 1d0))  ; the classic Rosenbrock starting point
      (learning-rate 0.001d0)
      (n-steps 2000))
  (dotimes (step n-steps)
    (let* ((tape (dm:make-tape))
           (vx (dm:make-var tape (aref point 0)))
           (vy (dm:make-var tape (aref point 1)))
           (vf (rosenbrock-var tape vx vy))
           (grad (dm:backward tape vf))
           (loss (dm:value vf)))
      (decf (aref point 0) (* learning-rate (aref grad (dm:index vx))))
      (decf (aref point 1) (* learning-rate (aref grad (dm:index vy))))
      (when (or (= step 0) (= step (1- n-steps)))
        (format t "step ~a: loss = ~a, x = [~a, ~a]~%" step loss (aref point 0) (aref point 1)))))
  (format t "(true minimum is at [1, 1] with loss 0)~%"))
