;; CMD: just TARGET=gis_ex run-lisp

;;;; ShapeLayer: write a tiny synthetic shapefile (.shp + .dbf) by hand, read it back and draw it as a map.
;;;; Real .shp/.dbf files are large binary bundles that don't belong in this repo, so this example
;;;; writes two "counties" (one plain square, one with a lake-shaped hole) using the same ESRI byte
;;;; layout ShapeLayer.read() expects, then reads them back through the public API.
(require "asdf")
(asdf:load-system "datamunge")

;;; Little-endian / big-endian packers that append to a byte buffer (like Lua's string.pack).
(defun make-buffer ()
  (make-array 0 :element-type '(unsigned-byte 8) :adjustable t :fill-pointer 0))

(defun pack-int (buf value nbytes &key big-endian)
  (let ((bytes (loop for i below nbytes collect (ldb (byte 8 (* 8 i)) value))))
    (dolist (b (if big-endian (reverse bytes) bytes))
      (vector-push-extend b buf))))

(defun pack-i4-le (buf value) (pack-int buf value 4))
(defun pack-i4-be (buf value) (pack-int buf value 4 :big-endian t))

(defun pack-double-le (buf x)
  (let ((x (float x 1d0)))
    (pack-int buf (sb-kernel:double-float-low-bits x) 4)
    (pack-int buf (sb-kernel:double-float-high-bits x) 4)))

(defun pack-bytes (buf bytes)
  (loop for b across bytes do (vector-push-extend b buf)))

(defun pack-ascii (buf string)
  (loop for ch across string do (vector-push-extend (char-code ch) buf)))

(defun pack-zeros (buf n)
  (dotimes (i n) (vector-push-extend 0 buf)))

(defun write-buffer (path buf)
  (with-open-file (f path :direction :output :element-type '(unsigned-byte 8) :if-exists :supersede)
    (write-sequence buf f)))

(defun ring-bounds (rings)
  "Bounding box (minx miny maxx maxy) of every point in RINGS."
  (let ((minx most-positive-double-float) (miny most-positive-double-float)
        (maxx most-negative-double-float) (maxy most-negative-double-float))
    (dolist (ring rings)
      (dolist (p ring)
        (destructuring-bind (x y) p
          (setf minx (min minx x) maxx (max maxx x)
                miny (min miny y) maxy (max maxy y)))))
    (list minx miny maxx maxy)))

(defun polygon-record (rings)
  "Bytes of one Polygon record's contents (without the record header)."
  (let ((points (apply #'append rings))
        (out (make-buffer)))
    (pack-i4-le out 5)                      ; shape type: Polygon
    (dolist (b (ring-bounds rings)) (pack-double-le out b))
    (pack-i4-le out (length rings))
    (pack-i4-le out (length points))
    (let ((start 0))
      (dolist (ring rings)
        (pack-i4-le out start)
        (incf start (length ring))))
    (dolist (p points)
      (pack-double-le out (first p))
      (pack-double-le out (second p)))
    out))

(defun write-counties-shp (path shapes)
  (let* ((contents (mapcar #'polygon-record shapes))
         (total-words (loop for c in contents sum (+ 4 (floor (length c) 2))))
         (f (make-buffer)))
    (pack-i4-be f 9994)
    (dotimes (i 5) (pack-i4-be f 0))
    (pack-i4-be f (+ 50 total-words))
    (pack-i4-le f 1000)
    (pack-i4-le f 5)                        ; Polygon
    (dolist (b (ring-bounds (apply #'append shapes))) (pack-double-le f b))
    (dotimes (i 4) (pack-double-le f 0d0))
    (loop for content in contents
          for i from 1
          do (pack-i4-be f i)
             (pack-i4-be f (floor (length content) 2))
             (pack-bytes f content))
    (write-buffer path f)))

(defun ljust (s n ch)
  (if (>= (length s) n)
      (subseq s 0 n)
      (concatenate 'string s (make-string (- n (length s)) :initial-element ch))))

(defun write-counties-dbf (path names populations)
  (let ((header-size (+ 32 (* 2 32) 1))
        (record-size (+ 1 12 8))
        (f (make-buffer)))
    (pack-bytes f #(#x03 0 0 0))
    (pack-int f (length names) 4)
    (pack-int f header-size 2)
    (pack-int f record-size 2)
    (pack-zeros f 20)
    (flet ((write-field (name field-type len)
             (pack-ascii f (ljust name 11 (code-char 0)))
             (pack-ascii f field-type)
             (pack-zeros f 4)
             (vector-push-extend len f)
             (pack-zeros f 15)))
      (write-field "NAME" "C" 12)
      (write-field "POP" "N" 8))
    (vector-push-extend #x0D f)
    (loop for name in names
          for pop in populations
          do (pack-ascii f " ")
             (pack-ascii f (ljust name 12 #\Space))
             (pack-ascii f (ljust pop 8 #\Space)))
    (write-buffer path f)))

(defparameter *base* "datamunge_gis_ex_counties_lisp")

;; Per the ESRI winding convention, outer rings are clockwise, holes counterclockwise.
(defparameter *plain* '(((10 0) (10 10) (20 10) (20 0) (10 0))))
(defparameter *with-lake* '(((0 0) (0 10) (10 10) (10 0) (0 0))
                            ((3 3) (4 3) (4 4) (3 4) (3 3))))

(write-counties-shp (format nil "~a.shp" *base*) (list *with-lake* *plain*))
(write-counties-dbf (format nil "~a.dbf" *base*) '("Lakeside" "Plainview") '("48231" "19876"))

(defparameter *counties* (dm:shape-layer-read *base*))

(format t "shapes = ~a, shape_type = ~a~%" (dm:size *counties*) (dm:shape-type *counties*))
(defun fmt-g (x)
  "Roughly C's %g: integral doubles print without a fractional part."
  (if (= x (ftruncate x)) (format nil "~d" (truncate x)) (format nil "~,6f" x)))

(format t "bounds = [~{~a~^, ~}]~%~%" (map 'list #'fmt-g (dm:bounds *counties*)))

(defparameter *attributes* (dm:attributes *counties*))
(format t "attributes~%~a~%~%" (dm:to-string *attributes*))

(dotimes (i (dm:size *counties*))
  (format t "~a: ~a ring(s)~%" (dm:string-at *attributes* "NAME" i) (dm:num-parts *counties* i)))

(let ((svg-path "datamunge_gis_ex_map_lisp.svg"))
  (dm:save-svg (dm:plot *counties*) svg-path)
  (format t "~%map saved to ~a~%" svg-path))
