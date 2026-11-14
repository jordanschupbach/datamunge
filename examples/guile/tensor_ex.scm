;; CMD: just TARGET=tensor_ex run-guile
;; Tensors: construction, shape ops, broadcasting, reductions, linear algebra, masks, iris.
(use-modules (oop goops) (datamunge) ((datamunge data) #:select (iris))
             ((srfi srfi-1) #:select (append-map)))

(define (shape-string tensor)
  (string-append "[" (string-join (map number->string (shape tensor)) ", ") "]"))

(display "=================== Construction ===================\n")
(format #t "zeros([2,3]): ~a\n" (to-string (tensor-zeros '(2 3))))
(format #t "eye(3): ~a\n" (to-string (tensor-eye 3)))

(define r (reshape (tensor-arange 0.0 12.0 1.0) '(3 4)))
(format #t "arange(0,12).reshape([3,4]): ~a\n" (to-string r))

(display "\n=================== Shape ops ===================\n")
(format #t "transpose -> shape ~a\n" (shape-string (transpose r)))
(format #t "slice(axis=1, start=1, stop=3): ~a\n" (to-string (slice r 1 1 3)))

(display "\n=================== Broadcasting arithmetic ===================\n")
(define col (tensor-from-values '(3 1) '(1.0 2.0 3.0)))
(define row (tensor-from-values '(1 4) '(10.0 20.0 30.0 40.0)))
(format #t "(3,1) + (1,4) -> ~a\n" (to-string (add col row)))

(display "\n=================== Reductions ===================\n")
(format #t "r.sum() = ~a, r.mean() = ~a\n" (sum r) (mean r))
(format #t "column means (axis=0): ~a\n" (to-string (mean-axis r 0)))

(display "\n=================== Linear algebra ===================\n")
(define a (tensor-from-values '(2 3) '(1.0 2.0 3.0 4.0 5.0 6.0)))
(define b (tensor-from-values '(3 2) '(7.0 8.0 9.0 10.0 11.0 12.0)))
(format #t "matmul(2x3, 3x2) -> ~a\n" (to-string (matmul a b)))
(define v1 (tensor-from-values '(3) '(1.0 2.0 3.0)))
(define v2 (tensor-from-values '(3) '(4.0 5.0 6.0)))
(format #t "dot([1,2,3], [4,5,6]) = ~a\n" (dot v1 v2))
(format #t "outer(v1, v2) -> ~a\n" (to-string (outer v1 v2)))

(display "\n=================== Comparisons & masks ===================\n")
(define mask (greater-equal r (tensor-full '(3 4) 6.0)))
(format #t "r >= 6 -> ~a\n" (to-string mask))
(format #t "count(r >= 6) = ~a\n" (sum mask))

(display "\n=================== A real dataset as a Tensor ===================\n")
(define iris-df (iris))
(define n (nrows iris-df))
(define columns '("Sepal.Length" "Sepal.Width" "Petal.Length" "Petal.Width"))
(define flat
  (append-map (lambda (i) (map (lambda (column) (numeric-at iris-df column i)) columns))
              (iota n)))
(define x (tensor-from-values (list n 4) flat))
(format #t "iris feature tensor shape: ~a\n" (shape-string x))

(define feature-means (mean-axis x 0))
(define centered (subtract x (reshape feature-means '(1 4))))
(define scatter (matmul (transpose centered) centered))
(format #t "feature means (Sepal.Length, Sepal.Width, Petal.Length, Petal.Width): ~a\n"
        (to-string feature-means))
(format #t "(X-mean)^T (X-mean) [4x4 scatter matrix]: ~a\n" (to-string scatter 16))

(define species (map (lambda (i) (string-at iris-df "Species" i)) (iota n)))
(define species-tensor (tensor-from-string-values (list n) species))
(define setosa-mask (equal species-tensor (tensor-from-string-values '(1) '("setosa"))))
(format #t "setosa count = ~a (of ~a rows)\n" (sum setosa-mask) n)
