;; CMD: just TARGET=knn_iris_ex run-guile
;; k-nearest-neighbours classifier on iris: leave-one-out accuracy and decision regions for several k.
(use-modules (oop goops) (datamunge) ((datamunge data) #:select (iris)))

(define iris-data (iris))
(format #t "iris: ~a rows x ~a cols\n\n" (nrows iris-data) (ncols iris-data))

(define model (make <knn-classifier> iris-data "Species ~ Petal.Length + Petal.Width"))
(print-summary model)

(format #t "\nConfusion matrix (leave-one-out; rows = actual, cols = predicted):\n~a\n"
        (to-string (confusion-matrix model)))

(display "\nLeave-one-out misclassified rows:\n")
(define fitted (predict model iris-data))
(define n (nrows iris-data))
(define misclassified 0)
(let loop ((i 0) (preds fitted))
  (when (< i n)
    (let ((actual (string-at iris-data "Species" i))
          (predicted (car preds)))
      (unless (string=? predicted actual)
        (set! misclassified (+ misclassified 1))
        (format #t "  row ~a: Petal.Length=~a Petal.Width=~a  actual=~a  predicted=~a\n"
                i (numeric-at iris-data "Petal.Length" i) (numeric-at iris-data "Petal.Width" i)
                actual predicted)))
    (loop (+ i 1) (cdr preds))))
(format #t "~a of ~a misclassified (~a%)\n" misclassified n (/ (* 100.0 misclassified) n))

(save (plot-decision-regions model "Petal.Length" "Petal.Width") "knn_iris_decision_regions_k5.svg")
(display "\nSaved knn_iris_decision_regions_k5.svg\n")

;; k=1 memorizes every training point exactly (jagged, overfit boundary with an island around
;; every point, including noise); k=25 averages over a much larger neighborhood (very smooth,
;; underfit boundary). (make <knn-classifier> data formula k metric weighted standardize)
(for-each
 (lambda (k)
   (let ((knn (make <knn-classifier> iris-data "Species ~ Petal.Length + Petal.Width" k))
         (file (format #f "knn_iris_decision_regions_k~a.svg" k)))
     (format #t "\nk=~2a leave-one-out accuracy: ~a%\n" k (* (training-accuracy knn) 100))
     (save (plot-decision-regions knn "Petal.Length" "Petal.Width") file)
     (format #t "Saved ~a\n" file)))
 '(1 25))
