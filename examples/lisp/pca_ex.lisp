;; CMD: just TARGET=pca_ex run-lisp

;;;; PCA on iris: summary, PC1 loadings, scores frame, score/scree plots, scaled vs unscaled.
(require "asdf")
(asdf:load-system "datamunge")

(defparameter *iris* (dm.data:iris))
(defparameter *feature-columns* #("Sepal.Length" "Sepal.Width" "Petal.Length" "Petal.Width"))

;; species labels, for the grouped (colored-by-species) scores plot
(defparameter *species*
  (coerce (loop for i below (dm:nrows *iris*) collect (dm:string-at *iris* "Species" i)) 'vector))

(format t "=================== PCA on iris (scaled) ===================~%")
(defparameter *pca* (dm:make-pca *iris* *feature-columns* t t))
(dm:print-summary *pca*)

(format t "~%PC1 loadings (which original features drive it):~%")
(loop for name across (dm:feature-names *pca*)
      for loading across (dm:component-loadings *pca* 0)
      do (format t "  ~a: ~,4f~%" name loading))

(format t "~%Training scores as a DataFrame:~%~a~%" (dm:to-string (dm:scores-frame *pca*)))

(dm:save-svg (dm:plot-scores-grouped *pca* *species*) "pca_iris_scores.svg")
(format t "~%Scores scatter (colored by species) saved as pca_iris_scores.svg~%")

(dm:save-svg (dm:plot-scree *pca*) "pca_iris_scree.svg")
(format t "Scree plot saved as pca_iris_scree.svg~%")

(format t "~%=================== PCA on iris (unscaled) ===================~%")
(let* ((unscaled (dm:make-pca *iris* *feature-columns* t nil))
       (scaled-ratio (dm:explained-variance-ratio *pca*))
       (unscaled-ratio (dm:explained-variance-ratio unscaled)))
  (format t "PC1 explains ~,2f% of variance (vs. ~,2f% scaled) -- Sepal.Length's larger raw variance ~
             dominates the unscaled covariance matrix.~%"
          (* 100 (aref unscaled-ratio 0)) (* 100 (aref scaled-ratio 0))))
