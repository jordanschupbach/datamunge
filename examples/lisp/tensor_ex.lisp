;; CMD: just TARGET=tensor_ex run-lisp

;;;; Tensors: construction, shape ops, broadcasting, reductions, linear algebra, masks, iris.
(require "asdf")
(asdf:load-system "datamunge")
(setf *read-default-float-format* (quote double-float)) ; print doubles without a d0 suffix

(defun shape-string (tensor)
  (format nil "[~{~a~^, ~}]" (coerce (dm:shape tensor) 'list)))

(format t "=================== Construction ===================~%")
(format t "zeros([2,3]): ~a~%" (dm:to-string (dm:tensor-zeros #(2 3))))
(format t "eye(3): ~a~%" (dm:to-string (dm:tensor-eye 3)))

(defparameter *r* (dm:reshape (dm:tensor-arange 0d0 12d0 1d0) #(3 4)))
(format t "arange(0,12).reshape([3,4]): ~a~%" (dm:to-string *r*))

(format t "~%=================== Shape ops ===================~%")
(format t "transpose -> shape ~a~%" (shape-string (dm:transpose *r*)))
(format t "slice(axis=1, start=1, stop=3): ~a~%" (dm:to-string (dm:slice *r* 1 1 3)))

(format t "~%=================== Broadcasting arithmetic ===================~%")
(let ((col (dm:tensor-from-values #(3 1) #(1d0 2d0 3d0)))
      (row (dm:tensor-from-values #(1 4) #(10d0 20d0 30d0 40d0))))
  (format t "(3,1) + (1,4) -> ~a~%" (dm:to-string (dm:add col row))))

(format t "~%=================== Reductions ===================~%")
(format t "r.sum() = ~a, r.mean() = ~a~%" (dm:sum *r*) (dm:mean *r*))
(format t "column means (axis=0): ~a~%" (dm:to-string (dm:mean-axis *r* 0)))

(format t "~%=================== Linear algebra ===================~%")
(let ((a (dm:tensor-from-values #(2 3) #(1d0 2d0 3d0 4d0 5d0 6d0)))
      (b (dm:tensor-from-values #(3 2) #(7d0 8d0 9d0 10d0 11d0 12d0))))
  (format t "matmul(2x3, 3x2) -> ~a~%" (dm:to-string (dm:matmul a b))))

(let ((v1 (dm:tensor-from-values #(3) #(1d0 2d0 3d0)))
      (v2 (dm:tensor-from-values #(3) #(4d0 5d0 6d0))))
  (format t "dot([1,2,3], [4,5,6]) = ~a~%" (dm:dot v1 v2))
  (format t "outer(v1, v2) -> ~a~%" (dm:to-string (dm:outer v1 v2))))

(format t "~%=================== Comparisons & masks ===================~%")
(let ((mask (dm:greater-equal *r* (dm:tensor-full #(3 4) 6d0))))
  (format t "r >= 6 -> ~a~%" (dm:to-string mask))
  (format t "count(r >= 6) = ~a~%" (dm:sum mask)))

(format t "~%=================== A real dataset as a Tensor ===================~%")
(let* ((iris (dm.data:iris))
       (n (dm:nrows iris))
       (flat (loop for i below n
                   nconc (loop for column in '("Sepal.Length" "Sepal.Width" "Petal.Length" "Petal.Width")
                               collect (dm:numeric-at iris column i))))
       (x (dm:tensor-from-values (vector n 4) flat)))
  (format t "iris feature tensor shape: ~a~%" (shape-string x))

  (let* ((feature-means (dm:mean-axis x 0))
         (centered (dm:subtract x (dm:reshape feature-means #(1 4))))
         (scatter (dm:matmul (dm:transpose centered) centered)))
    (format t "feature means (Sepal.Length, Sepal.Width, Petal.Length, Petal.Width): ~a~%"
            (dm:to-string feature-means))
    (format t "(X-mean)^T (X-mean) [4x4 scatter matrix]: ~a~%" (dm:to-string scatter 16)))

  (let* ((species (loop for i below n collect (dm:string-at iris "Species" i)))
         (species-tensor (dm:tensor-from-string-values (vector n) species))
         (setosa-mask (dm:equal species-tensor (dm:tensor-from-string-values #(1) '("setosa")))))
    (format t "setosa count = ~a (of ~a rows)~%" (dm:sum setosa-mask) n)))
