;; CMD: just TARGET=naive_bayes_penguins_ex run-guile
;; Naive Bayes on penguins: mixed numeric/categorical predictors, confusion matrix, misclassified rows, plots.
(use-modules (oop goops) (datamunge) ((datamunge data) #:select (penguins)))

;; Note: <naive-bayes-classifier> drops incomplete rows for its own formula columns internally
;; when fitting, matching every other formula-based model in this library.
(define penguins-data (penguins))
(format #t "penguins: ~a rows x ~a cols\n\n" (nrows penguins-data) (ncols penguins-data))

;; A mix of numeric (Gaussian likelihood) and categorical (frequency-table likelihood)
;; predictors in one formula -- each modeled independently given the class, per the naive
;; Bayes assumption.
(define model (make <naive-bayes-classifier> penguins-data
                "species ~ bill_length_mm + bill_depth_mm + island + sex"))
(print-summary model)

(format #t "\nConfusion matrix (rows = actual, cols = predicted):\n~a\n" (to-string (confusion-matrix model)))

(display "\nMisclassified rows:\n")
(define predictions (predict model penguins-data))
(define columns '("species" "bill_length_mm" "bill_depth_mm" "island" "sex"))
(define misclassified 0)
(let loop ((i 0) (preds predictions))
  (when (< i (nrows penguins-data))
    (unless (or-map (lambda (c) (null? penguins-data c i)) columns)
      (let ((actual (string-at penguins-data "species" i))
            (predicted (car preds)))
        (unless (string=? predicted actual)
          (set! misclassified (+ misclassified 1))
          (format #t "  row ~a: bill_length=~a bill_depth=~a island=~a sex=~a  actual=~a  predicted=~a\n"
                  i (numeric-at penguins-data "bill_length_mm" i) (numeric-at penguins-data "bill_depth_mm" i)
                  (string-at penguins-data "island" i) (string-at penguins-data "sex" i)
                  actual predicted))))
    (loop (+ i 1) (cdr preds))))
(format #t "~a misclassified (of ~a rows, some incomplete)\n" misclassified (nrows penguins-data))

;; plot-decision-regions requires exactly two NUMERIC predictors, so build a separate
;; two-predictor model (bill measurements alone) just for visualization.
(define bill-only (make <naive-bayes-classifier> penguins-data "species ~ bill_length_mm + bill_depth_mm"))
(format #t "\nbill-measurements-only model training accuracy: ~a%\n" (* 100 (training-accuracy bill-only)))
(save (plot-classification bill-only penguins-data "bill_length_mm" "bill_depth_mm")
      "naive_bayes_penguins_classification.svg")
(save (plot-decision-regions bill-only "bill_length_mm" "bill_depth_mm")
      "naive_bayes_penguins_decision_regions.svg")
(display "Saved naive_bayes_penguins_classification.svg and naive_bayes_penguins_decision_regions.svg\n")
