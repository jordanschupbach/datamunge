;; Demonstrates ShapeLayer: reading a shapefile (.shp geometry + .dbf attributes) and drawing it
;; as a map. Real .shp/.dbf files are large binary bundles that don't belong in this repo, so this
;; example first writes a tiny synthetic shapefile by hand (two "counties": one plain square, one
;; with a lake-shaped hole) using the same ESRI byte layout ShapeLayer.read() expects, then reads
;; it back through the public API. Guile's (rnrs bytevectors) mirrors Python's struct.pack for the
;; binary layout.
(use-modules (datamunge)
             (ice-9 format)
             (rnrs bytevectors)
             (rnrs io ports))

;; --- little/big-endian packing helpers ------------------------------------------------------
(define (i32le n)
  (let ((bv (make-bytevector 4 0))) (bytevector-u32-set! bv 0 n (endianness little)) bv))
(define (i32be n)
  (let ((bv (make-bytevector 4 0))) (bytevector-u32-set! bv 0 n (endianness big)) bv))
(define (u16le n)
  (let ((bv (make-bytevector 2 0))) (bytevector-u16-set! bv 0 n (endianness little)) bv))
(define (f64le x)
  (let ((bv (make-bytevector 8 0)))
    (bytevector-ieee-double-set! bv 0 (exact->inexact x) (endianness little)) bv))

(define (bv-concat bvs)
  (let* ((total (apply + (map bytevector-length bvs)))
         (out (make-bytevector total 0)))
    (let loop ((lst bvs) (off 0))
      (if (null? lst)
          out
          (let* ((b (car lst)) (len (bytevector-length b)))
            (bytevector-copy! b 0 out off len)
            (loop (cdr lst) (+ off len)))))))

;; Left-justify string `s` into `n` bytes, truncating if longer, padding with `pad-byte`.
(define (ljust-bytes s n pad-byte)
  (let* ((b (string->utf8 s)) (len (bytevector-length b)))
    (if (>= len n)
        (let ((out (make-bytevector n 0))) (bytevector-copy! b 0 out 0 n) out)
        (let ((out (make-bytevector n pad-byte))) (bytevector-copy! b 0 out 0 len) out))))

;; --- .shp writer ----------------------------------------------------------------------------
;; A point is (list x y); a ring is a list of points; a shape is a list of rings.
(define (polygon-record rings)
  (let* ((all-points (apply append rings))
         (px (map car all-points))
         (py (map cadr all-points))
         (minx (apply min px)) (maxx (apply max px))
         (miny (apply min py)) (maxy (apply max py))
         (parts '())
         (start 0))
    (for-each (lambda (ring)
                (set! parts (cons (i32le start) parts))
                (set! start (+ start (length ring))))
              rings)
    (set! parts (reverse parts))
    (bv-concat
     (append
      (list (i32le 5) ; shape type: Polygon
            (f64le minx) (f64le miny) (f64le maxx) (f64le maxy)
            (i32le (length rings))
            (i32le (length all-points)))
      parts
      (apply append
             (map (lambda (ring)
                    (apply append
                           (map (lambda (p) (list (f64le (car p)) (f64le (cadr p)))) ring)))
                  rings))))))

(define (write-counties-shp path shapes)
  (let* ((contents (map polygon-record shapes))
         (total-words (apply + (map (lambda (c) (+ 4 (quotient (bytevector-length c) 2))) contents)))
         (all-points (apply append (apply append shapes)))
         (px (map car all-points)) (py (map cadr all-points))
         (minx (apply min px)) (maxx (apply max px))
         (miny (apply min py)) (maxy (apply max py))
         (port (open-file-output-port path (file-options no-fail))))
    (put-bytevector port (i32be 9994))
    (put-bytevector port (bv-concat (list (i32be 0) (i32be 0) (i32be 0) (i32be 0) (i32be 0))))
    (put-bytevector port (i32be (+ 50 total-words)))
    (put-bytevector port (i32le 1000))
    (put-bytevector port (i32le 5)) ; Polygon
    (put-bytevector port (bv-concat (list (f64le minx) (f64le miny) (f64le maxx) (f64le maxy))))
    (put-bytevector port (bv-concat (list (f64le 0.0) (f64le 0.0) (f64le 0.0) (f64le 0.0))))
    (let loop ((cs contents) (i 1))
      (when (not (null? cs))
        (let ((content (car cs)))
          (put-bytevector port (i32be i))
          (put-bytevector port (i32be (quotient (bytevector-length content) 2)))
          (put-bytevector port content)
          (loop (cdr cs) (+ i 1)))))
    (close-port port)))

;; --- .dbf writer ----------------------------------------------------------------------------
(define (write-field port name field-type length)
  (put-bytevector port (ljust-bytes name 11 0))
  (put-bytevector port (string->utf8 field-type))
  (put-bytevector port (make-bytevector 4 0))
  (put-bytevector port (u8-list->bytevector (list length)))
  (put-bytevector port (make-bytevector 15 0)))

(define (write-counties-dbf path names populations)
  (let ((header-size 97)   ; 32 + 2*32 + 1
        (record-size 21)   ; 1 + 12 + 8
        (port (open-file-output-port path (file-options no-fail))))
    (put-bytevector port (u8-list->bytevector (list 3 0 0 0)))
    (put-bytevector port (i32le (length names)))
    (put-bytevector port (u16le header-size))
    (put-bytevector port (u16le record-size))
    (put-bytevector port (make-bytevector 20 0))
    (write-field port "NAME" "C" 12)
    (write-field port "POP" "N" 8)
    (put-bytevector port (u8-list->bytevector (list #x0D)))
    (let loop ((ns names) (ps populations))
      (when (not (null? ns))
        (put-bytevector port (string->utf8 " "))
        (put-bytevector port (ljust-bytes (car ns) 12 (char->integer #\space)))
        (put-bytevector port (ljust-bytes (car ps) 8 (char->integer #\space)))
        (loop (cdr ns) (cdr ps))))
    (close-port port)))

;; --- build the synthetic shapefile & read it back -------------------------------------------
;; Per the ESRI winding convention, outer rings are clockwise, holes counterclockwise.
(define plain
  (list (list (list 10 0) (list 10 10) (list 20 10) (list 20 0) (list 10 0))))
(define with-lake
  (list (list (list 0 0) (list 0 10) (list 10 10) (list 10 0) (list 0 0))
        (list (list 3 3) (list 4 3) (list 4 4) (list 3 4) (list 3 3))))

(define base "datamunge_gis_ex_counties_guile")

(write-counties-shp (string-append base ".shp") (list with-lake plain))
(write-counties-dbf (string-append base ".dbf") (list "Lakeside" "Plainview") (list "48231" "19876"))

(define counties (ShapeLayer-read base))

(format #t "shapes = ~a, shape_type = ~a\n"
        (ShapeLayer-size counties) (ShapeLayer-shape-type counties))
(define bounds (ShapeLayer-bounds counties))
(format #t "bounds = [~a]\n"
        (string-join (map number->string (vector->list bounds)) ", "))
(newline)

(define attributes (ShapeLayer-attributes counties))
(format #t "attributes\n")
(format #t "~a\n" (DataFrame-to-string attributes))
(newline)

(do ((i 0 (+ i 1))) ((= i (ShapeLayer-size counties)))
  (format #t "~a: ~a ring(s)\n"
          (DataFrame-string-at attributes "NAME" i) (ShapeLayer-num-parts counties i)))

(define svg-path "datamunge_gis_ex_map_guile.svg")
(Plot-save-svg (ShapeLayer-plot counties) svg-path)
(format #t "\nmap saved to ~a\n" svg-path)
