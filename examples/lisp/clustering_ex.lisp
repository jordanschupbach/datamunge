;; CMD: just TARGET=clustering_ex run-lisp

;;;; Clustering iris: k-means, agglomerative (several linkages, dendrogram re-cut) and DBSCAN.
(require "asdf")
(asdf:load-system "datamunge")

(defun report-purity (cluster-labels species n-clusters)
  "For each cluster, print its size and its majority species."
  (let ((votes (make-array n-clusters)))
    (dotimes (c n-clusters) (setf (aref votes c) (make-hash-table :test #'equal)))
    (loop for label across cluster-labels
          for sp across species
          when (>= label 0)
            do (incf (gethash sp (aref votes label) 0)))
    (dotimes (c n-clusters)
      (let ((total 0) (best-species nil) (best -1))
        (maphash (lambda (sp n)
                   (incf total n)
                   (when (> n best) (setf best-species sp best n)))
                 (aref votes c))
        (when (> total 0)
          (format t "  cluster ~a: ~a points, majority ~a (~a/~a)~%"
                  c total best-species best total))))))

(defparameter *iris* (dm.data:iris))
(defparameter *species*
  (coerce (loop for i below (dm:nrows *iris*) collect (dm:string-at *iris* "Species" i)) 'vector))
(defparameter *feature-columns* #("Sepal.Length" "Sepal.Width" "Petal.Length" "Petal.Width"))

(format t "=================== K-means (k=3) on iris ===================~%")
(let ((kmeans (dm:make-k-means *iris* *feature-columns* 3)))
  (format t "Inertia: ~f, iterations (best run): ~a~%"
          (dm:inertia kmeans) (dm:iterations-used kmeans))
  (report-purity (dm:labels kmeans) *species* 3))

(format t "~%=================== Agglomerative clustering on iris ===================~%")
(dolist (linkage '("ward" "complete" "average"))
  (let ((model (dm:make-agglomerative-clustering *iris* *feature-columns* 3 linkage)))
    (format t "--- linkage=~a ---~%" linkage)
    (report-purity (dm:labels model) *species* 3)))

(format t "~%--- Re-cutting the ward dendrogram at k=2 without refitting ---~%")
(let ((ward-model (dm:make-agglomerative-clustering *iris* *feature-columns* 3 "ward")))
  (report-purity (dm:cut ward-model 2) *species* 2))

(format t "~%=================== DBSCAN on iris ===================~%")
(let ((dbscan (dm:make-dbscan *iris* *feature-columns* 0.6d0 5)))
  (format t "Clusters found: ~a, noise points: ~a (of ~a)~%"
          (dm:n-clusters dbscan) (dm:n-noise dbscan) (dm:observations dbscan))
  (report-purity (dm:labels dbscan) *species* (dm:n-clusters dbscan)))
