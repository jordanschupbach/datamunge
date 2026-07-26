(use-modules (datamunge) (ice-9 format))

(define iris (DataFrame-iris))
(define features (list "Sepal.Length" "Sepal.Width" "Petal.Length" "Petal.Width"))

;; species labels, for the grouped (colored-by-species) scores plot
(define n (DataFrame-nrows iris))
(define species '())
(do ((i 0 (+ i 1))) ((= i n))
  (set! species (cons (DataFrame-string-at iris "Species" i) species)))
(set! species (reverse species))

(format #t "=================== PCA on iris (scaled) ===================\n")
(define pca (new-PCA iris features #t #t))
(PCA-print-summary pca)

(format #t "\nPC1 loadings (which original features drive it):\n")
(define names (PCA-feature-names pca))
(define loadings (PCA-component-loadings pca 0))
(do ((i 0 (+ i 1))) ((= i (vector-length names)))
  (format #t "  ~a: ~,4f\n" (vector-ref names i) (vector-ref loadings i)))

(format #t "\nTraining scores as a DataFrame:\n")
(format #t "~a\n" (DataFrame-to-string (PCA-scores-frame pca)))

(define scatter (PCA-plot-scores-grouped pca species))
(Plot-save-svg scatter "pca_iris_scores.svg")
(format #t "\nScores scatter (colored by species) saved as pca_iris_scores.svg\n")

(define scree (PCA-plot-scree pca))
(Plot-save-svg scree "pca_iris_scree.svg")
(format #t "Scree plot saved as pca_iris_scree.svg\n")

(format #t "\n=================== PCA on iris (unscaled) ===================\n")
(define unscaled (new-PCA iris features #t #f))
(define scaled-ratio (PCA-explained-variance-ratio pca))
(define unscaled-ratio (PCA-explained-variance-ratio unscaled))
(format #t
        (string-append
         "PC1 explains ~,2f% of variance (vs. ~,2f% scaled) -- Sepal.Length's larger raw "
         "variance dominates the unscaled covariance matrix.\n")
        (* (vector-ref unscaled-ratio 0) 100)
        (* (vector-ref scaled-ratio 0) 100))
