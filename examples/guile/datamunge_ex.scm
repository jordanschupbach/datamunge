(use-modules (datamunge))

(display "Hello from Guile + Datamunge\n")
(hello)

(let ((v (make_dvector 10.0 20.0 30.0)))
  (format #t "sum_dvector = ~a\n" (sum_dvector v)))

(let ((p (make_dpair 4.0 5.0)))
  (format #t "sum_dpair = ~a\n" (sum_dpair p)))

