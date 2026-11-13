;; CMD: just TARGET=stl_ex run-lisp

;;;; C++ std::vector and std::pair values in Lisp: vectors are 0-based Lisp vectors, pairs are lists.
(require "asdf")
(asdf:load-system "datamunge")

(setf *read-default-float-format* 'double-float)

(let* ((n 100)
       (v (make-array n :element-type 'double-float)))
  (dotimes (i n) (setf (aref v i) (* i 1.5d0)))
  (loop for x across v do (format t "~a~%" x))

  (let ((v2 (make-array n :element-type 'integer)))
    (dotimes (i n) (setf (aref v2 i) (floor (* i 1.5d0))))
    (loop for x across v2 do (format t "~a~%" x))))

(let ((p (list 3 4)))
  (format t "p: (~a, ~a)~%" (first p) (second p)))

(let ((p2 (list 10 20)))
  (format t "p2: (~a, ~a)~%" (first p2) (second p2)))
