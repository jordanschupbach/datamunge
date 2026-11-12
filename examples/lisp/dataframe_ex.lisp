;;;; DataFrame basics: nulls, drop_duplicates, fill_null, select/sort, group-by, join and summaries.
(require "asdf")
(asdf:load-system "datamunge")

(defparameter *sales*
  (let ((frame (dm:make-data-frame)))
    (dm:add-string-column frame "region" #("west" "west" "east" "south" "south" "south"))
    (dm:add-string-column frame "product" #("widget" "widget" "widget" "gizmo" "gizmo" "gizmo"))
    (dm:add-numeric-column frame "sales" #(10d0 10d0 14d0 8d0 0d0 11d0) #(1 1 1 1 0 1))
    (dm:add-string-column frame "quarter" #("Q1" "Q1" "Q1" "Q2" "Q2" "") #(1 1 1 1 1 0))
    frame))

(format t "raw data~%~a~%~%" (dm:to-string *sales*))

(defparameter *cleaned* (dm:drop-duplicates *sales* #("region" "product" "sales" "quarter")))
(dm:fill-null-string *cleaned* "quarter" "unknown")
(dm:fill-null-numeric *cleaned* "sales" 0d0)
(format t "after drop_duplicates + fill_null~%~a~%~%" (dm:to-string *cleaned*))

(let ((selected (dm:sort-by (dm:select *cleaned* #("region" "sales" "quarter")) "sales" nil)))
  (format t "selected + sorted~%~a~%~%" (dm:to-string selected)))

(defparameter *grouped*
  (dm:sort-by (dm:group-by-sum *cleaned* #("region") #("sales")) "sales" nil))
(format t "group_by_sum(region)~%~a~%~%" (dm:to-string *grouped*))

(let ((targets (dm:make-data-frame)))
  (dm:add-string-column targets "region" #("west" "east" "south"))
  (dm:add-numeric-column targets "target" #(18d0 12d0 25d0))
  (format t "joined with targets~%~a~%~%"
          (dm:to-string (dm:join *grouped* targets "region" "region" "left"))))

(let ((shape (dm:shape *cleaned*)))
  (format t "shape = (~a, ~a)~%" (elt shape 0) (elt shape 1)))
(format t "sales count = ~a~%" (dm:numeric-count *cleaned* "sales"))
(format t "sales nulls = ~a~%" (dm:numeric-null-count *cleaned* "sales"))
(format t "sales sum = ~f~%" (dm:numeric-sum *cleaned* "sales"))
(format t "sales mean = ~f~%" (dm:numeric-mean *cleaned* "sales"))
