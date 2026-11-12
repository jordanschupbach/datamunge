;;;; ODEs: built-in named systems, harmonic-oscillator energy check and a Lorenz phase plot.
(require "asdf")
(asdf:load-system "datamunge")

(defun fmt-g (x)
  "Approximate C's %g: six significant digits, no padding."
  (string-trim " " (format nil "~,6g" x)))

(defun last-state (sol) (dm:state-at sol (1- (dm:size sol))))

;; NOTE: like the Lua version this example uses the built-in named systems via solve-builtin.

(format t "=================== Every named built-in system (no director needed) ===================~%")
(let ((solver (dm:make-ode-solver)))
  (dolist (s (list (list "exponential_decay" #(1d0) #(1d0))
                   (list "logistic_growth" #(1d0 1d0) #(0.5d0))
                   (list "harmonic_oscillator" #(1d0) #(1d0 0d0))
                   (list "van_der_pol" #(1d0) #(2d0 0d0))
                   (list "lorenz" (vector 10d0 28d0 (/ 8d0 3d0)) #(1d0 1d0 1d0))))
    (destructuring-bind (name params y0) s
      (let ((sol (dm:solve-builtin solver name params y0 0d0 1d0)))
        (format t "~22a steps=~6a final state=[~{~a~^, ~}]~%"
                name (dm:steps-taken sol) (map 'list #'fmt-g (last-state sol)))))))

(defun solution-series (sol &rest components)
  "Return the time vector followed by one vector per requested state component."
  (let ((n (dm:size sol)))
    (apply #'values
           (coerce (loop for i below n collect (dm:time-at sol i)) 'vector)
           (loop for c in components
                 collect (coerce (loop for i below n collect (aref (dm:state-at sol i) c))
                                 'vector)))))

(format t "~%=================== Harmonic oscillator (energy conservation) ===================~%")
(let* ((options (dm:make-ode-options))
       (solver (progn (setf (dm:method options) :rk4
                            (dm:step-size options) 0.01d0)
                      (dm:make-ode-solver options)))
       (sol (dm:solve-builtin solver "harmonic_oscillator" #(1d0) #(1d0 0d0) 0d0 20d0))
       (final (last-state sol))
       (x (aref final 0))
       (v (aref final 1)))
  (format t "x(20) = ~,6f (cos(20) = ~,6f)~%" x (cos 20d0))
  (format t "energy x^2+v^2 = ~,6f (should stay near 1.0)~%" (+ (* x x) (* v v)))
  (multiple-value-bind (ts xs vs) (solution-series sol 0 1)
    (let ((plot (dm:r-plot-plot ts xs "l" "x(t)")))
      (dm:lines plot ts vs "v(t)")
      (dm:y-label (dm:x-label (dm:title plot "Harmonic Oscillator") "t") "state")
      (dm:save-svg plot "ode_harmonic_oscillator_lisp.svg")
      (format t "wrote ode_harmonic_oscillator_lisp.svg~%"))))

(format t "~%=================== Lorenz attractor (phase plane) ===================~%")
(let* ((options (dm:make-ode-options))
       (solver (progn (setf (dm:method options) :rk4
                            (dm:step-size options) 0.005d0)
                      (dm:make-ode-solver options)))
       (sol (dm:solve-builtin solver "lorenz" (vector 10d0 28d0 (/ 8d0 3d0)) #(1d0 1d0 1d0)
                              0d0 25d0)))
  (format t "steps_taken = ~d~%" (dm:steps-taken sol))
  (multiple-value-bind (ts xs zs) (solution-series sol 0 2)
    (declare (ignore ts))
    (let ((plot (dm:r-plot-plot xs zs "l" "trajectory")))
      (dm:y-label (dm:x-label (dm:title plot "Lorenz Attractor (x-z phase plane)") "x") "z")
      (dm:save-svg plot "ode_lorenz_phase_plane_lisp.svg")
      (format t "wrote ode_lorenz_phase_plane_lisp.svg~%"))))
