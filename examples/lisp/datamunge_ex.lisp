;; CMD: just TARGET=datamunge_ex run-lisp

;;;; Smoke test of the basic bindings: vectors, pairs and a native Callback object.
(require "asdf")
(asdf:load-system "datamunge")

(dm:hello)

(let ((v (dm:make-dvector 1d0 2d0 3d0)))
  (format t "sum_dvector:~c~f~%" #\Tab (dm:sum-dvector v)))

(let ((p (dm:make-dpair 1.25d0 2.75d0)))
  (format t "sum_dpair:~c~f~%" #\Tab (dm:sum-dpair p)))

(let ((cb (dm:make-callback)))
  (format t "call_with_callback(3.0):~c~f~%" #\Tab (dm:call-with-callback 3d0 cb))
  (let ((v2 (dm:map-dvector-with-callback (dm:make-dvector 1d0 2d0 3d0) cb)))
    (format t "sum_dvector(map_dvector_with_callback(1,2,3)):~c~f~%" #\Tab (dm:sum-dvector v2))))
