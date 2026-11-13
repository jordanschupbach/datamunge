;; CMD: just TARGET=mds_ex run-lisp

;;;; Classical multidimensional scaling of iris: euclidean embedding, plot, and a manhattan comparison.
(require "asdf")
(asdf:load-system "datamunge")

(defparameter *iris* (dm.data:iris))
(defparameter *feature-columns* #("Sepal.Length" "Sepal.Width" "Petal.Length" "Petal.Width"))
(defparameter *species*
  (let ((v (make-array (dm:nrows *iris*))))
    (dotimes (i (length v) v)
      (setf (aref v i) (dm:string-at *iris* "Species" i)))))

(format t "=================== Classical MDS on iris (euclidean) ===================~%")
(defparameter *mds* (dm:make-mds *iris* *feature-columns* 2 "euclidean"))
(dm:print-summary *mds*)

(format t "~%Embedding as a DataFrame:~%~a~%" (dm:to-string (dm:embedding-frame *mds*)))

(dm:save-svg (dm:plot-embedding-grouped *mds* *species*) "mds_iris_embedding.svg")
(format t "~%Embedding scatter (colored by species) saved as mds_iris_embedding.svg~%")

(format t "~%=================== Classical MDS on iris (manhattan) ===================~%")
(let ((manhattan-mds (dm:make-mds *iris* *feature-columns* 2 "manhattan")))
  (format t "Goodness of fit: euclidean=~,4f, manhattan=~,4f~%"
          (dm:goodness-of-fit *mds*) (dm:goodness-of-fit manhattan-mds)))
