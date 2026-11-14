;; CMD: just TARGET=hypothesis_tests_ex run-guile
;; Classical hypothesis tests (t, Wilcoxon, ANOVA, correlation, F, normality, chi-squared, binomial) on iris.
(use-modules (oop goops) (datamunge) ((datamunge data) #:select (iris)) ((srfi srfi-1) #:select (filter-map)))

(define (print-result label r)
  (let ((alt (case (alternative r) ((less) "less") ((greater) "greater") (else "two.sided"))))
    (display (string-append
              label ": statistic=" (number->string (statistic r))
              (if (zero? (parameter1 r)) "" (format #f ", df1=~a" (parameter1 r)))
              (if (zero? (parameter2 r)) "" (format #f ", df2=~a" (parameter2 r)))
              (format #f ", p=~a (~a)" (p-value r) alt)
              (if (has-conf-int r)
                  (format #f ", CI=[~a, ~a]" (conf-int-lower r) (conf-int-upper r))
                  "")
              " -- " (method_ r) "\n"))))

;; The values of `col` in df, optionally only for rows of the given species.
(define* (column df col #:optional species)
  (filter-map (lambda (i)
                (and (or (not species) (string=? (string-at df "Species" i) species))
                     (numeric-at df col i)))
              (iota (nrows df))))

(define iris-df (iris))

(define setosa-petal (column iris-df "Petal.Length" "setosa"))
(define versicolor-petal (column iris-df "Petal.Length" "versicolor"))
(define virginica-petal (column iris-df "Petal.Length" "virginica"))

(display "=================== t-tests: petal length, setosa vs. versicolor ===================\n")
(print-result "Welch two-sample t-test" (t-test-two-sample setosa-petal versicolor-petal))
(print-result "Wilcoxon rank-sum test" (wilcoxon-rank-sum-test setosa-petal versicolor-petal))

(display "\n=================== One-way ANOVA / Kruskal-Wallis across all 3 species ===================\n")
(define all-petal (append setosa-petal versicolor-petal virginica-petal))
(define sizes (map length (list setosa-petal versicolor-petal virginica-petal)))
(print-result "One-way ANOVA" (one-way-anova all-petal sizes))
(print-result "Kruskal-Wallis" (kruskal-wallis-test all-petal sizes))

(display "\n=================== Correlation: sepal length vs. petal length ===================\n")
(define sepal-length (column iris-df "Sepal.Length"))
(define petal-length (column iris-df "Petal.Length"))
(print-result "Pearson correlation" (pearson-correlation-test sepal-length petal-length))
(print-result "Spearman correlation" (spearman-correlation-test sepal-length petal-length))

(display "\n=================== F-test: petal length variance, setosa vs. virginica ===================\n")
(print-result "F test" (f-test-variance setosa-petal virginica-petal))

(display "\n=================== Normality: is sepal length normally distributed within setosa? ===================\n")
(define setosa-sepal (column iris-df "Sepal.Length" "setosa"))
(print-result "Shapiro-Francia" (shapiro-francia-test setosa-sepal))
(print-result "KS vs. fitted normal" (ks-test-one-sample-normal setosa-sepal 5.006 0.3525))

(display "\n=================== Chi-squared / Fisher: is petal length \"long\" independent of species? ===================\n")
(define median-all (list-ref (sort all-petal <) (quotient (length all-petal) 2)))
(define (count-long v) (length (filter (lambda (x) (> x median-all)) v)))
(define setosa-long (count-long setosa-petal))
(define setosa-short (- (length setosa-petal) setosa-long))
(define versicolor-long (count-long versicolor-petal))
(define versicolor-short (- (length versicolor-petal) versicolor-long))
(format #t "table: setosa=[~a,~a] versicolor=[~a,~a]\n"
        setosa-long setosa-short versicolor-long versicolor-short)
(define table (map exact->inexact (list setosa-long setosa-short versicolor-long versicolor-short)))
(print-result "Chi-squared independence" (chi-squared-test-independence table 2 2))
(print-result "Fisher's exact test"
              (fisher-exact-test-2x2 setosa-long setosa-short versicolor-long versicolor-short))

(display "\n=================== Proportion / binomial: fraction of \"long\" petals overall ===================\n")
(define long-count (count-long all-petal))
(define n (length all-petal))
(print-result "One-sample proportion test (vs 0.5)" (proportion-test-one-sample long-count n 0.5))
(print-result "Exact binomial test (vs 0.5)" (binomial-test long-count n 0.5))
