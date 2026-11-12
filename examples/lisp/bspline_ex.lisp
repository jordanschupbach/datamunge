;;;; Tensor-product B-spline regression: fit z ~ bs(x, y) to a smooth surface and predict.
(require "asdf")
(asdf:load-system "datamunge")

;; Sample a smooth surface z = sin(x) + y^2 on an 8x8 grid over [0, 1]^2.
(defparameter *data*
  (let (x y z)
    (dotimes (iy 8)
      (dotimes (ix 8)
        (let ((xv (/ ix 7d0)) (yv (/ iy 7d0)))
          (push xv x)
          (push yv y)
          (push (+ (sin xv) (* yv yv)) z))))
    (let ((frame (dm:make-data-frame)))
      (dm:add-numeric-column frame "x" (nreverse x))
      (dm:add-numeric-column frame "y" (nreverse y))
      (dm:add-numeric-column frame "z" (nreverse z))
      frame)))

(defparameter *surface* (dm:make-lm *data* "z ~ bs(x, y)"))
(format t "Fitted z ~~ bs(x, y)~%")
(dm:print-summary *surface*)

(let ((new-points (dm:make-data-frame)))
  (dm:add-numeric-column new-points "x" #(0.25d0 0.75d0))
  (dm:add-numeric-column new-points "y" #(0.50d0 0.25d0))
  (let ((pred (dm:predict *surface* new-points)))
    (format t "Predictions: [~,6f, ~,6f]~%" (aref pred 0) (aref pred 1))))
