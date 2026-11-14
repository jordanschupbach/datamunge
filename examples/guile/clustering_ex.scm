;; CMD: just TARGET=clustering_ex run-guile
;; Clustering iris: k-means, agglomerative (several linkages, dendrogram re-cut) and DBSCAN.
(use-modules (oop goops) (datamunge) ((datamunge data) #:select (iris)))

(define iris-data (iris))
(define species (map (lambda (i) (string-at iris-data "Species" i)) (iota (nrows iris-data))))
(define feature-columns '("Sepal.Length" "Sepal.Width" "Petal.Length" "Petal.Width"))

;; For each cluster, print its size and its majority species (negative labels are noise).
(define (report-purity cluster-labels n-clusters)
  ;; votes[c] is an alist of (species . count), kept in first-seen order.
  (define votes (make-vector n-clusters '()))
  (for-each
   (lambda (label sp)
     (when (>= label 0)
       (let* ((counts (vector-ref votes label))
              (entry (assoc sp counts)))
         (if entry
             (set-cdr! entry (+ (cdr entry) 1))
             (vector-set! votes label (append counts (list (cons sp 1))))))))
   cluster-labels species)
  (do ((c 0 (+ c 1))) ((= c n-clusters))
    (let* ((counts (vector-ref votes c))
           (total (apply + (map cdr counts))))
      (unless (zero? total)
        (let ((best (car counts)))
          (for-each (lambda (e) (when (> (cdr e) (cdr best)) (set! best e))) counts)
          (format #t "  cluster ~a: ~a points, majority ~a (~a/~a)\n"
                  c total (car best) (cdr best) total))))))

(display "=================== K-means (k=3) on iris ===================\n")
(define kmeans (make <k-means> iris-data feature-columns 3))
(format #t "Inertia: ~a, iterations (best run): ~a\n" (inertia kmeans) (iterations-used kmeans))
(report-purity (labels kmeans) 3)

(display "\n=================== Agglomerative clustering on iris ===================\n")
(for-each
 (lambda (linkage)
   (let ((model (make <agglomerative-clustering> iris-data feature-columns 3 linkage)))
     (format #t "--- linkage=~a ---\n" linkage)
     (report-purity (labels model) 3)))
 '("ward" "complete" "average"))

(display "\n--- Re-cutting the ward dendrogram at k=2 without refitting ---\n")
(define ward-model (make <agglomerative-clustering> iris-data feature-columns 3 "ward"))
(report-purity (cut ward-model 2) 2)

(display "\n=================== DBSCAN on iris ===================\n")
(define dbscan (make <dbscan> iris-data feature-columns 0.6 5))
(format #t "Clusters found: ~a, noise points: ~a (of ~a)\n"
        (n-clusters dbscan) (n-noise dbscan) (observations dbscan))
(report-purity (labels dbscan) (n-clusters dbscan))
