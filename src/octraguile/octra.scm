(define-module (octra)
  #:export (hello
            make_dvector
            sum_dvector
            make_dpair
            sum_dpair))

;; Load the compiled SWIG extension into this module, so its symbols land here.
;; Avoid doing this at compile time so auto-compilation doesn't fail before the
;; extension is available on GUILE_EXTENSION_PATH.
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
           (let ((candidate (string-append d "/octra.so")))
             (if (false-if-exception (access? candidate F_OK))
                 (try-load candidate)
                 (loop rest))))))))

  (unless (or (try-load "octra")
              (try-load "liboctra")
              (try-load "octra_guile")
              (try-load "liboctra_guile")
              (try-load-from-extension-path))
    (error "Could not load octra Guile extension (octra.so). Set GUILE_EXTENSION_PATH.")))

;; SWIG's Guile backend uses Scheme-style names (hyphens) for C identifiers with
;; underscores. Provide underscore aliases so the API matches the other bindings.
(define make_dvector make-dvector)
(define sum_dvector sum-dvector)
(define make_dpair make-dpair)
(define sum_dpair sum-dpair)
