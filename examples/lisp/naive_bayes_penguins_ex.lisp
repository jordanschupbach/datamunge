;; CMD: just TARGET=naive_bayes_penguins_ex run-lisp

;;;; Naive Bayes on penguins: mixed numeric/categorical predictors, confusion matrix, misclassified rows, plots.
(require "asdf")
(asdf:load-system "datamunge")

;; Print doubles as 39.1 rather than 39.1d0.
(setf *read-default-float-format* 'double-float)

;; Note: NaiveBayesClassifier drops incomplete rows for its own formula columns internally
;; when fitting, matching every other formula-based model in this library.
(defparameter *penguins* (dm.data:penguins))
(format t "penguins: ~a rows x ~a cols~%~%" (dm:nrows *penguins*) (dm:ncols *penguins*))

;; A mix of numeric (Gaussian likelihood) and categorical (frequency-table likelihood)
;; predictors in one formula -- each modeled independently given the class, per the naive
;; Bayes assumption.
(defparameter *model*
  (dm:make-naive-bayes-classifier *penguins* "species ~ bill_length_mm + bill_depth_mm + island + sex"))
(dm:print-summary *model*)

(format t "~%Confusion matrix (rows = actual, cols = predicted):~%~a~%"
        (dm:to-string (dm:confusion-matrix *model*)))

(format t "~%Misclassified rows:~%")
(let ((predictions (dm:predict *model* *penguins*))
      (misclassified 0)
      (columns '("species" "bill_length_mm" "bill_depth_mm" "island" "sex")))
  (dotimes (i (dm:nrows *penguins*))
    (unless (some (lambda (column) (dm:is-null *penguins* column i)) columns)
      (let ((actual (dm:string-at *penguins* "species" i))
            (predicted (aref predictions i)))
        (unless (string= predicted actual)
          (incf misclassified)
          (format t "  row ~a: bill_length=~a bill_depth=~a island=~a sex=~a  actual=~a  predicted=~a~%"
                  i (dm:numeric-at *penguins* "bill_length_mm" i)
                  (dm:numeric-at *penguins* "bill_depth_mm" i)
                  (dm:string-at *penguins* "island" i) (dm:string-at *penguins* "sex" i)
                  actual predicted)))))
  (format t "~a misclassified (of ~a rows, some incomplete)~%" misclassified (dm:nrows *penguins*)))

;; plot_decision_regions requires exactly two NUMERIC predictors, so build a separate
;; two-predictor model (bill measurements alone) just for visualization.
(let ((bill-only (dm:make-naive-bayes-classifier *penguins* "species ~ bill_length_mm + bill_depth_mm")))
  (format t "~%bill-measurements-only model training accuracy: ~a%~%"
          (* 100d0 (dm:training-accuracy bill-only)))
  (dm:save (dm:plot-classification bill-only *penguins* "bill_length_mm" "bill_depth_mm")
           "naive_bayes_penguins_classification.svg")
  (dm:save (dm:plot-decision-regions bill-only "bill_length_mm" "bill_depth_mm")
           "naive_bayes_penguins_decision_regions.svg")
  (format t "Saved naive_bayes_penguins_classification.svg and naive_bayes_penguins_decision_regions.svg~%"))
