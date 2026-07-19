(use-modules (datamunge))

(define n 100)
(define v (new-DVector))
(do ((i 0 (+ i 1))) ((= i n))
  (DVector-push! v (* i 1.5)))
(do ((i 0 (+ i 1))) ((= i n))
  (format #t "~a\n" (DVector-ref v i)))

(define v2 (new-IVector))
(do ((i 0 (+ i 1))) ((= i n))
  (IVector-push! v2 (inexact->exact (truncate (* i 1.5)))))
(do ((i 0 (+ i 1))) ((= i n))
  (format #t "~a\n" (IVector-ref v2 i)))

(define p (new-IPair 3 4))
(format #t "p: (~a, ~a)\n" (IPair-first-get p) (IPair-second-get p))

(define p2 (new-DPair 10.0 20.0))
(format #t "p2: (~a, ~a)\n" (DPair-first-get p2) (DPair-second-get p2))
