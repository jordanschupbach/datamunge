;;;; Classical hypothesis tests (t, Wilcoxon, ANOVA, correlation, F, normality, chi-squared, binomial) on iris.
(require "asdf")
(asdf:load-system "datamunge")

(setf *read-default-float-format* 'double-float)

(defparameter *alt-names* '(:less "less" :greater "greater" :two-sided "two.sided"))

(defun print-result (label r)
  (format t "~a: statistic=~a" label (dm:statistic r))
  (unless (zerop (dm:parameter1 r)) (format t ", df1=~a" (dm:parameter1 r)))
  (unless (zerop (dm:parameter2 r)) (format t ", df2=~a" (dm:parameter2 r)))
  (format t ", p=~a (~a)" (dm:p-value r) (getf *alt-names* (dm:alternative r)))
  (when (dm:has-conf-int r)
    (format t ", CI=[~a, ~a]" (dm:conf-int-lower r) (dm:conf-int-upper r)))
  (format t " -- ~a~%" (dm:method r)))

(defun column-for-species (df column species)
  (coerce (loop for i below (dm:nrows df)
                when (string= (dm:string-at df "Species" i) species)
                  collect (dm:numeric-at df column i))
          '(vector double-float)))

(defparameter *iris* (dm:data-frame-iris))

(defparameter *setosa-petal* (column-for-species *iris* "Petal.Length" "setosa"))
(defparameter *versicolor-petal* (column-for-species *iris* "Petal.Length" "versicolor"))
(defparameter *virginica-petal* (column-for-species *iris* "Petal.Length" "virginica"))

(format t "=================== t-tests: petal length, setosa vs. versicolor ===================~%")
(print-result "Welch two-sample t-test" (dm:t-test-two-sample *setosa-petal* *versicolor-petal*))
(print-result "Wilcoxon rank-sum test" (dm:wilcoxon-rank-sum-test *setosa-petal* *versicolor-petal*))

(format t "~%=================== One-way ANOVA / Kruskal-Wallis across all 3 species ===================~%")
(defparameter *all-petal* (concatenate '(vector double-float) *setosa-petal* *versicolor-petal* *virginica-petal*))
(let ((sizes (vector (length *setosa-petal*) (length *versicolor-petal*) (length *virginica-petal*))))
  (print-result "One-way ANOVA" (dm:one-way-anova *all-petal* sizes))
  (print-result "Kruskal-Wallis" (dm:kruskal-wallis-test *all-petal* sizes)))

(format t "~%=================== Correlation: sepal length vs. petal length ===================~%")
(let ((sepal-length (loop for i below (dm:nrows *iris*) collect (dm:numeric-at *iris* "Sepal.Length" i)))
      (petal-length (loop for i below (dm:nrows *iris*) collect (dm:numeric-at *iris* "Petal.Length" i))))
  (print-result "Pearson correlation" (dm:pearson-correlation-test sepal-length petal-length))
  (print-result "Spearman correlation" (dm:spearman-correlation-test sepal-length petal-length)))

(format t "~%=================== F-test: petal length variance, setosa vs. virginica ===================~%")
(print-result "F test" (dm:f-test-variance *setosa-petal* *virginica-petal*))

(format t "~%=================== Normality: is sepal length normally distributed within setosa? ===================~%")
(let ((setosa-sepal (column-for-species *iris* "Sepal.Length" "setosa")))
  (print-result "Shapiro-Francia" (dm:shapiro-francia-test setosa-sepal))
  (print-result "KS vs. fitted normal" (dm:ks-test-one-sample-normal setosa-sepal 5.006d0 0.3525d0)))

(format t "~%=================== Chi-squared / Fisher: is petal length \"long\" independent of species? ===================~%")
(defparameter *median-all*
  (let ((sorted (sort (copy-seq *all-petal*) #'<)))
    (aref sorted (floor (length sorted) 2))))
(defun count-long (v) (count-if (lambda (x) (> x *median-all*)) v))
(let* ((setosa-long (count-long *setosa-petal*))
       (setosa-short (- (length *setosa-petal*) setosa-long))
       (versicolor-long (count-long *versicolor-petal*))
       (versicolor-short (- (length *versicolor-petal*) versicolor-long)))
  (format t "table: setosa=[~a,~a] versicolor=[~a,~a]~%" setosa-long setosa-short versicolor-long versicolor-short)
  (let ((table-vec (map '(vector double-float) (lambda (n) (float n 1d0))
                        (list setosa-long setosa-short versicolor-long versicolor-short))))
    (print-result "Chi-squared independence" (dm:chi-squared-test-independence table-vec 2 2)))
  (print-result "Fisher's exact test"
                (dm:fisher-exact-test-2x2 setosa-long setosa-short versicolor-long versicolor-short)))

(format t "~%=================== Proportion / binomial: fraction of \"long\" petals overall ===================~%")
(let ((long-count (count-long *all-petal*))
      (n (length *all-petal*)))
  (print-result "One-sample proportion test (vs 0.5)" (dm:proportion-test-one-sample long-count n 0.5d0))
  (print-result "Exact binomial test (vs 0.5)" (dm:binomial-test long-count n 0.5d0)))
