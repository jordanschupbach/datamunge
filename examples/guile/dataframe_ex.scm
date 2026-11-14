;; CMD: just TARGET=dataframe_ex run-guile
;; DataFrame basics: nulls, drop_duplicates, fill_null, select/sort, group-by, join and summaries.
(use-modules (oop goops) (datamunge))

(define sales (make <data-frame>))
(add-string-column sales "region" '("west" "west" "east" "south" "south" "south"))
(add-string-column sales "product" '("widget" "widget" "widget" "gizmo" "gizmo" "gizmo"))
(add-numeric-column sales "sales" '(10.0 10.0 14.0 8.0 0.0 11.0) '(1 1 1 1 0 1))
(add-string-column sales "quarter" '("Q1" "Q1" "Q1" "Q2" "Q2" "") '(1 1 1 1 1 0))

(format #t "raw data\n~a\n\n" (to-string sales))

(define cleaned (drop-duplicates sales '("region" "product" "sales" "quarter")))
(fill-null-string cleaned "quarter" "unknown")
(fill-null-numeric cleaned "sales" 0.0)
(format #t "after drop_duplicates + fill_null\n~a\n\n" (to-string cleaned))

(define selected (sort-by (select cleaned '("region" "sales" "quarter")) "sales" #f))
(format #t "selected + sorted\n~a\n\n" (to-string selected))

(define grouped (sort-by (group-by-sum cleaned '("region") '("sales")) "sales" #f))
(format #t "group_by_sum(region)\n~a\n\n" (to-string grouped))

(define targets (make <data-frame>))
(add-string-column targets "region" '("west" "east" "south"))
(add-numeric-column targets "target" '(18.0 12.0 25.0))
(format #t "joined with targets\n~a\n\n" (to-string (join grouped targets "region" "region" "left")))

(define dims (shape cleaned))
(format #t "shape = (~a, ~a)\n" (car dims) (cadr dims))
(format #t "sales count = ~a\n" (numeric-count cleaned "sales"))
(format #t "sales nulls = ~a\n" (numeric-null-count cleaned "sales"))
(format #t "sales sum = ~a\n" (numeric-sum cleaned "sales"))
(format #t "sales mean = ~a\n" (numeric-mean cleaned "sales"))
