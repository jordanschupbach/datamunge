;;;; Time series: ARIMA, seasonal ARIMA and exponential smoothing (SES, Holt, Holt-Winters).
(require "asdf")
(asdf:load-system "datamunge")
(setf *read-default-float-format* 'double-float) ; print doubles without a d0 suffix

(defvar *rng* (sb-ext:seed-random-state 42))

(defun seed (n) (setf *rng* (sb-ext:seed-random-state n)))

(defun uniform ()
  ;; (0, 1]: avoids log(0) in the Box-Muller transform.
  (- 1d0 (random 1d0 *rng*)))

(defun gauss ()
  (let ((u1 (uniform)) (u2 (uniform)))
    (* (sqrt (* -2d0 (log u1))) (cos (* 2d0 pi u2)))))

(defun join (vec)
  (format nil "~{~a~^, ~}" (coerce vec 'list)))

(format t "=================== ARIMA(1,1,1) on a simulated random walk with drift ===================~%")
(seed 42)
(defparameter *y*
  (let ((level 100d0) (prev-shock 0d0) (out '()))
    (dotimes (i 150 (nreverse out))
      (let ((shock (gauss)))
        ;; integrated ARMA(0,1) increments with drift
        (incf level (+ 0.3d0 (* 0.4d0 prev-shock) shock))
        (push level out)
        (setf prev-shock shock)))))

(defparameter *model*
  (let ((options (dm:make-arima-options)))
    (setf (dm:p options) 1
          (dm:d options) 1
          (dm:q options) 1
          (dm:de-population-size options) 80
          (dm:de-max-generations options) 400)
    (dm:make-arima *y* options)))

(format t "AR coefficient: ~a~%" (aref (dm:ar-coefficients *model*) 0))
(format t "MA coefficient: ~a~%" (aref (dm:ma-coefficients *model*) 0))
(format t "sigma^2: ~a, AIC: ~a, BIC: ~a~%" (dm:sigma2 *model*) (dm:aic *model*) (dm:bic *model*))

(destructuring-bind (forecast std-errors) (dm:forecast-with-intervals *model* 6)
  (format t "6-step forecast: ~a~%" (join forecast))
  (format t "forecast std. errors: ~a~%" (join std-errors)))

(format t "~%=================== SARIMA(1,0,0)(1,1,0)_12 on a seasonal series ===================~%")
(seed 7)
(defparameter *s*
  (let ((prev 0d0) (out '()))
    (dotimes (i 120 (nreverse out))
      (setf prev (+ (* 0.5d0 prev) (gauss)))
      (push (+ 20d0 (* 0.2d0 i) (* 5d0 (sin (/ (* 2d0 pi i) 12d0))) prev) out))))

(defparameter *model2*
  (let ((options (dm:make-arima-options)))
    (setf (dm:p options) 1
          (dm:seasonal-p options) 1
          (dm:seasonal-d options) 1
          (dm:seasonal-period options) 12
          (dm:de-population-size options) 100
          (dm:de-max-generations options) 500)
    (dm:make-arima *s* options)))
(format t "AR coefficient: ~a, seasonal AR coefficient: ~a~%"
        (aref (dm:ar-coefficients *model2*) 0)
        (aref (dm:seasonal-ar-coefficients *model2*) 0))
(format t "12-step forecast: ~a~%" (join (dm:forecast *model2* 12)))

(format t "~%=================== Exponential smoothing: Holt-Winters on retail-style seasonal data ===================~%")
(defparameter *seasonal-shape* #(0.8d0 0.75d0 0.9d0 0.95d0 1d0 1.05d0 1.1d0 1.05d0 1d0 1.1d0 1.3d0 1.6d0))
(seed 11)
(defparameter *y2*
  (loop for i from 0 below 48
        collect (+ (* (+ 100d0 (* 2d0 i)) (aref *seasonal-shape* (mod i 12)))
                   (* (gauss) 3d0))))

(defparameter *es-model*
  (let ((options (dm:make-exponential-smoothing-options)))
    (setf (dm:trend options) :additive
          (dm:seasonal options) :multiplicative
          (dm:seasonal-period options) 12
          (dm:de-population-size options) 60
          (dm:de-max-generations options) 300)
    (dm:make-exponential-smoothing *y2* options)))
(format t "alpha=~a beta=~a gamma=~a~%"
        (dm:alpha *es-model*) (dm:beta *es-model*) (dm:gamma *es-model*))
(format t "sigma^2: ~a, AIC: ~a~%" (dm:sigma2 *es-model*) (dm:aic *es-model*))
(format t "12-month forecast: ~a~%" (join (dm:forecast *es-model* 12)))

(format t "~%=================== Simple exponential smoothing vs. Holt's linear trend ===================~%")
(defparameter *flat* #(50.2d0 49.8d0 50.5d0 49.6d0 50.1d0 50.3d0 49.9d0 50d0 50.4d0 49.7d0))

(let ((ses (dm:make-exponential-smoothing *flat* (dm:make-exponential-smoothing-options))))
  (format t "SES alpha= ~a~%" (dm:alpha ses))
  (format t "SES 5-step forecast: ~a~%" (join (dm:forecast ses 5))))

(let* ((holt-options (dm:make-exponential-smoothing-options))
       (holt (progn (setf (dm:trend holt-options) :additive)
                    (dm:make-exponential-smoothing *flat* holt-options))))
  (format t "Holt alpha=~a beta=~a~%" (dm:alpha holt) (dm:beta holt))
  (format t "Holt 5-step forecast: ~a~%" (join (dm:forecast holt 5))))
