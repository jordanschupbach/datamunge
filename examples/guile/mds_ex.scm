(use-modules (datamunge) (ice-9 format))

(define iris (DataFrame-iris))
(define features (list "Sepal.Length" "Sepal.Width" "Petal.Length" "Petal.Width"))

(define n (DataFrame-nrows iris))
(define species '())
(do ((i 0 (+ i 1))) ((= i n))
  (set! species (cons (DataFrame-string-at iris "Species" i) species)))
(set! species (reverse species))

(format #t "=================== Classical MDS on iris (euclidean) ===================\n")
(define mds (new-MDS iris features 2 "euclidean"))
(MDS-print-summary mds)

(format #t "\nEmbedding as a DataFrame:\n")
(format #t "~a\n" (DataFrame-to-string (MDS-embedding-frame mds)))

(define scatter (MDS-plot-embedding-grouped mds species))
(Plot-save-svg scatter "mds_iris_embedding.svg")
(format #t "\nEmbedding scatter (colored by species) saved as mds_iris_embedding.svg\n")

(format #t "\n=================== Classical MDS on iris (manhattan) ===================\n")
(define manhattan-mds (new-MDS iris features 2 "manhattan"))
(format #t "Goodness of fit: euclidean=~,4f, manhattan=~,4f\n"
        (MDS-goodness-of-fit mds) (MDS-goodness-of-fit manhattan-mds))
