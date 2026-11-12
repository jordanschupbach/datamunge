;;;; Smoke tests of the datamunge Common Lisp system (just test-lisp).
(require "asdf")
(asdf:load-system "datamunge")

(defvar *checks* 0)
(defmacro check (form &optional (message (format nil "~s" form)))
  `(progn (incf *checks*)
          (unless ,form (error "check failed: ~a" ,message))))

;; Smoke test: call into the native library.
(dm:hello)

;; std::vector<double> and std::pair<double,double> round trips.
(let ((v (dm:make-dvector 1d0 2d0 3d0)))
  (check (typep v '(vector double-float)))
  (check (equalp v #(1d0 2d0 3d0)))
  (check (= (dm:sum-dvector v) 6d0))
  (check (= (dm:sum-dvector '(0.5d0 1.5d0)) 2d0) "lists convert to std::vector"))
(let ((p (dm:make-dpair 1.25d0 2.75d0)))
  (check (equal p '(1.25d0 2.75d0)))
  (check (= (dm:sum-dpair p) 4d0)))

;; The C++ Callback class, and a Lisp subclass overriding its virtual call() (a director).
(let ((cb (dm:make-callback)))
  (check (= (dm:call-with-callback 3d0 cb) 3d0))
  (check (= (dm:sum-dvector (dm:map-dvector-with-callback (dm:make-dvector 1d0 2d0 3d0) cb)) 6d0)))

(defclass doubler (dm:callback) ())
(dm:define-override dm:call ((self doubler) x) (* 2 x))
(let ((doubler (make-instance 'doubler)))
  (check (= (dm:call-with-callback 3d0 doubler) 6d0) "the Lisp override runs")
  (check (= (dm:sum-dvector (dm:map-dvector-with-callback '(1d0 2d0 3d0) doubler)) 12d0)))

;; A Lisp error inside an override reaches the caller as the original condition.
(define-condition from-lisp (error) ())
(defclass failing (dm:callback) ())
(dm:define-override dm:call ((self failing) x) (declare (ignore x)) (error 'from-lisp))
(check (eq :signalled (handler-case (dm:call-with-callback 1d0 (make-instance 'failing))
                        (from-lisp () :signalled)))
       "the Lisp condition propagates")

;; C++ exceptions become dm:native-error.
(check (typep (handler-case (dm:seql 0d0 1d0 -1) (dm:native-error (e) e)) 'dm:native-error)
       "C++ exceptions signal dm:native-error")

;; Docstrings carry the C++ documentation.
(check (search "evenly spaced" (documentation 'dm:seql 'function)) "docstrings")

(format t "test_datamunge.lisp: all ~d checks passed~%" *checks*)
