;; CMD: just TARGET=datamunge_ex run-guile
;; Smoke test of the basic bindings: vectors, pairs and a native Callback object.
(use-modules (oop goops) (datamunge))

(hello)

(define v (make-dvector 1.0 2.0 3.0))
(format #t "sum_dvector:\t~a\n" (sum-dvector v))

(define p (make-dpair 1.25 2.75))
(format #t "sum_dpair:\t~a\n" (sum-dpair p))

(define cb (make <callback>))
(format #t "call_with_callback(3.0):\t~a\n" (call-with-callback 3.0 cb))
(define v2 (map-dvector-with-callback (make-dvector 1.0 2.0 3.0) cb))
(format #t "sum_dvector(map_dvector_with_callback(1,2,3)):\t~a\n" (sum-dvector v2))
(close! cb)
