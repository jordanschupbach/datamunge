;; CMD: just TARGET=mds_ex run-guile
;; Classical multidimensional scaling of iris: euclidean embedding, plot, and a manhattan comparison.
(use-modules (oop goops) (datamunge) ((datamunge data) #:select (iris)) (ice-9 format))

(define iris-df (iris))
(define feature-columns '("Sepal.Length" "Sepal.Width" "Petal.Length" "Petal.Width"))
(define species
  (map (lambda (i) (string-at iris-df "Species" i)) (iota (nrows iris-df))))

(display "=================== Classical MDS on iris (euclidean) ===================\n")
(define mds (make <mds> iris-df feature-columns 2 "euclidean"))
(print-summary mds)

(format #t "\nEmbedding as a DataFrame:\n~a\n" (to-string (embedding-frame mds)))

(save-svg (plot-embedding-grouped mds species) "mds_iris_embedding.svg")
(display "\nEmbedding scatter (colored by species) saved as mds_iris_embedding.svg\n")

(display "\n=================== Classical MDS on iris (manhattan) ===================\n")
(define manhattan-mds (make <mds> iris-df feature-columns 2 "manhattan"))
(format #t "Goodness of fit: euclidean=~,4f, manhattan=~,4f\n"
        (goodness-of-fit mds) (goodness-of-fit manhattan-mds))
