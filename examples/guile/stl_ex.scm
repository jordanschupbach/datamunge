;; CMD: just TARGET=stl_ex run-guile
;; C++ std::vector and std::pair values in Guile: vectors are lists, pairs are Scheme pairs.
(use-modules (oop goops) (datamunge))

(define n 100)
(define v (map (lambda (i) (* i 1.5)) (iota n)))
(for-each (lambda (x) (display x) (newline)) v)
(define v2 (map (lambda (i) (inexact->exact (floor (* i 1.5)))) (iota n)))
(for-each (lambda (x) (display x) (newline)) v2)

(define p '(3 . 4))
(format #t "p: (~a, ~a)\n" (car p) (cdr p))

(define p2 '(10 . 20))
(format #t "p2: (~a, ~a)\n" (car p2) (cdr p2))

;; A C++ std::pair<double,double> comes back as a Scheme pair.
(define p3 (make-dpair 1.25 2.75))
(format #t "make_dpair(1.25, 2.75): (~a, ~a), sum_dpair = ~a\n" (car p3) (cdr p3) (sum-dpair p3))
