;; CMD: just TARGET=pca_ex run-guile
;; PCA on iris: summary, PC1 loadings, scores frame, score/scree plots, scaled vs unscaled.
(use-modules (oop goops) (datamunge) ((datamunge data) #:select (iris)) (ice-9 format))

(define iris-df (iris))
(define feature-columns '("Sepal.Length" "Sepal.Width" "Petal.Length" "Petal.Width"))

;; species labels, for the grouped (colored-by-species) scores plot
(define species
  (map (lambda (i) (string-at iris-df "Species" i)) (iota (nrows iris-df))))

(display "=================== PCA on iris (scaled) ===================\n")
(define pca (make <pca> iris-df feature-columns #t #t))
(print-summary pca)

(display "\nPC1 loadings (which original features drive it):\n")
(for-each (lambda (name loading) (format #t "  ~a: ~,4f\n" name loading))
          (feature-names pca) (component-loadings pca 0))

(format #t "\nTraining scores as a DataFrame:\n~a\n" (to-string (scores-frame pca)))

(save-svg (plot-scores-grouped pca species) "pca_iris_scores.svg")
(display "\nScores scatter (colored by species) saved as pca_iris_scores.svg\n")

(save-svg (plot-scree pca) "pca_iris_scree.svg")
(display "Scree plot saved as pca_iris_scree.svg\n")

(display "\n=================== PCA on iris (unscaled) ===================\n")
(define unscaled (make <pca> iris-df feature-columns #t #f))
(format #t "PC1 explains ~,2f% of variance (vs. ~,2f% scaled) -- Sepal.Length's larger raw variance dominates the unscaled covariance matrix.\n"
        (* 100 (car (explained-variance-ratio unscaled)))
        (* 100 (car (explained-variance-ratio pca))))
