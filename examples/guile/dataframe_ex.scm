(use-modules (datamunge))

(define sales (DataFrame-empty))
(DataFrame-add-string-column sales "region" (list "west" "west" "east" "south" "south" "south"))
(DataFrame-add-string-column sales "product" (list "widget" "widget" "widget" "gizmo" "gizmo" "gizmo"))
(DataFrame-add-numeric-column sales "sales" (list 10.0 10.0 14.0 8.0 0.0 11.0) (list 1 1 1 1 0 1))
(DataFrame-add-string-column sales "quarter" (list "Q1" "Q1" "Q1" "Q2" "Q2" "") (list 1 1 1 1 1 0))

(format #t "raw data\n")
(format #t "~a\n\n" (DataFrame-to-string sales))

(define cleaned (DataFrame-drop-duplicates sales (list "region" "product" "sales" "quarter")))
(DataFrame-fill-null-string cleaned "quarter" "unknown")
(DataFrame-fill-null-numeric cleaned "sales" 0.0)
(format #t "after drop_duplicates + fill_null\n")
(format #t "~a\n\n" (DataFrame-to-string cleaned))

(define selected (DataFrame-sort-by (DataFrame-select cleaned (list "region" "sales" "quarter")) "sales" #f))
(format #t "selected + sorted\n")
(format #t "~a\n\n" (DataFrame-to-string selected))

(define grouped (DataFrame-sort-by (DataFrame-group-by-sum cleaned (list "region") (list "sales")) "sales" #f))
(format #t "group_by_sum(region)\n")
(format #t "~a\n\n" (DataFrame-to-string grouped))

(define targets (DataFrame-empty))
(DataFrame-add-string-column targets "region" (list "west" "east" "south"))
(DataFrame-add-numeric-column targets "target" (list 18.0 12.0 25.0))
(define joined (DataFrame-join grouped targets "region" "region" #t))
(format #t "joined with targets\n")
(format #t "~a\n\n" (DataFrame-to-string joined))

(define shape (DataFrame-shape cleaned))
(format #t "shape = (~a, ~a)\n" (vector-ref shape 0) (vector-ref shape 1))
(format #t "sales count = ~a\n" (DataFrame-numeric-count cleaned "sales"))
(format #t "sales nulls = ~a\n" (DataFrame-numeric-null-count cleaned "sales"))
(format #t "sales sum = ~a\n" (DataFrame-numeric-sum cleaned "sales"))
(format #t "sales mean = ~a\n" (DataFrame-numeric-mean cleaned "sales"))
