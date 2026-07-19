(define-module (datamunge))

;; Load the compiled SWIG extension into this module, so its symbols land here.
;; Avoid doing this at compile time so auto-compilation doesn't fail before the
;; extension is available on GUILE_EXTENSION_PATH.
;;
;; Previously this module hand-maintained a 5-symbol #:export list (hello,
;; make_dvector, sum_dvector, make_dpair, sum_dpair) -- the SAME bug found and fixed
;; in datamungetcl's pkgIndex.tcl: load-extension defines every bound symbol
;; directly in this module, but #:export controls what's visible to *users* of
;; (use-modules (datamunge)), so the rest of the API (DataFrame-iris, new-KMeans,
;; etc.) existed but was unreachable from outside. Fixed the same way as Tcl: snapshot
;; this module's local bindings before load-extension, then export every symbol that
;; wasn't there before (see datamunge_tcl_bindings.md).
(eval-when (load eval)
  (use-modules (ice-9 match))

  (define (try-load name)
    (false-if-exception (load-extension name "SWIG_init")))

  (define (split-colon s)
    (if (or (not s) (string=? s ""))
        '()
        (let loop ((start 0) (parts '()))
          (let ((idx (string-index s #\: start)))
            (if idx
                (loop (+ idx 1)
                      (cons (substring s start idx) parts))
                (reverse (cons (substring s start (string-length s)) parts)))))))

  (define (try-load-from-extension-path)
    (let* ((ext-path (getenv "GUILE_EXTENSION_PATH"))
           (dirs (split-colon ext-path)))
      (let loop ((ds dirs))
        (match ds
          (() #f)
          ((d . rest)
           (let ((candidate (string-append d "/datamunge.so")))
             (if (false-if-exception (access? candidate F_OK))
                 (try-load candidate)
                 (loop rest))))))))

  (define %before-load-symbols (module-map (lambda (sym var) sym) (current-module)))

  (unless (or (try-load "datamunge")
              (try-load "libdatamunge")
              (try-load "datamunge_guile")
              (try-load "libdatamunge_guile")
              (try-load-from-extension-path))
    (error "Could not load datamunge Guile extension (datamunge.so). Set GUILE_EXTENSION_PATH."))

  (module-export! (current-module)
                  (filter (lambda (sym) (not (memq sym %before-load-symbols)))
                          (module-map (lambda (sym var) sym) (current-module))))

  ;; SWIG's Guile backend uses Scheme-style names (hyphens) for C identifiers with
  ;; underscores. Keep these underscore aliases for backward compatibility with the
  ;; pre-existing datamunge_ex.scm example; new code should use the native hyphenated
  ;; names (make-dvector, sum-dvector, etc.) like every other bound symbol.
  (define make_dvector make-dvector)
  (define sum_dvector sum-dvector)
  (define make_dpair make-dpair)
  (define sum_dpair sum-dpair)
  (module-export! (current-module) '(make_dvector sum_dvector make_dpair sum_dpair)))
