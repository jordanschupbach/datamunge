(use-modules (octra))

(hello)

(let ((v (make_dvector 1.0 2.0 3.0)))
  (unless (= 6.0 (sum_dvector v))
    (error "sum_dvector mismatch" v (sum_dvector v))))

(let ((p (make_dpair 1.0 2.0)))
  (unless (= 3.0 (sum_dpair p))
    (error "sum_dpair mismatch" p (sum_dpair p))))

(display "guile bindings ok\n")

