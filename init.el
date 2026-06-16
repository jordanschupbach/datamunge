;;; init.el --- Batch Org export for datamunge -*- lexical-binding: t; -*-

;; Usage:
;;   emacs --batch -Q -l init.el -- examples/org/dense_linear_algebra.org build/org/dense_linear_algebra.html
;;   emacs --batch -Q -l init.el -- examples/org/dense_linear_algebra.org build/org/dense_linear_algebra.md
;;
;; Behavior:
;; - Applies the nearest `.envrc` before executing Babel blocks or exporting.
;; - Disables Org Babel confirmation prompts for batch execution.
;; - Infers the export backend from the output file extension.

(setq package-enable-at-startup nil)

(defconst datamunge-repo-root
  (file-name-directory (or load-file-name buffer-file-name default-directory))
  "Repository root for this batch Org configuration.")

(setq user-emacs-directory
      (expand-file-name ".cache/emacs/" datamunge-repo-root))

(make-directory user-emacs-directory t)

(require 'org)
(require 'ob)
(require 'ox)
(require 'ox-ascii)
(require 'ox-html)
(require 'ox-latex)
(require 'ox-md)
(require 'ox-odt)
(require 'seq)

(setq org-confirm-babel-evaluate nil)
(setq org-export-use-babel t)

(defvar datamunge--babel-languages nil)

(defun datamunge--enable-babel-language (lang feature)
  "Enable Babel LANG when FEATURE can be loaded."
  (when (require feature nil 'noerror)
    (push (cons lang t) datamunge--babel-languages)))

(dolist (entry '((emacs-lisp . ob-emacs-lisp)
                 (shell . ob-shell)
                 (C . ob-C)
                 (python . ob-python)
                 (js . ob-js)
                 (R . ob-R)))
  (datamunge--enable-babel-language (car entry) (cdr entry)))

(org-babel-do-load-languages
 'org-babel-load-languages
 (nreverse datamunge--babel-languages))

(defun datamunge--ensure-parent-directory (path)
  "Create the parent directory for PATH if needed."
  (let ((parent (file-name-directory (expand-file-name path))))
    (unless (file-directory-p parent)
      (make-directory parent t))))

(defun datamunge--direnv-root (start-dir)
  "Find the nearest parent of START-DIR containing `.envrc`."
  (let ((dir (file-name-as-directory (expand-file-name start-dir)))
        (prev nil)
        (found nil))
    (while (and dir (not (equal dir prev)) (not found))
      (when (file-exists-p (expand-file-name ".envrc" dir))
        (setq found dir))
      (setq prev dir)
      (setq dir (file-name-directory (directory-file-name dir))))
    found))

(defun datamunge--apply-direnv-exec (workdir)
  "Apply the evaluated direnv environment for WORKDIR."
  (let ((direnv (executable-find "direnv"))
        (root (datamunge--direnv-root workdir)))
    (when (and direnv root)
      (with-temp-buffer
        (let ((status (call-process direnv nil t nil "exec" root "env" "-0")))
          (when (zerop status)
            (dolist (entry (split-string (buffer-string) "\0" t))
              (let ((eq-pos (string-search "=" entry)))
                (when eq-pos
                  (setenv (substring entry 0 eq-pos)
                          (substring entry (1+ eq-pos))))))
            (let ((path (getenv "PATH")))
              (when path
                (setq exec-path
                      (append (parse-colon-path path)
                              (list exec-directory)))))))))))

(defun datamunge--apply-direnv-environment (workdir)
  "Apply the nearest direnv environment for WORKDIR."
  (datamunge--apply-direnv-exec workdir))

(defun datamunge--backend-for-output (output)
  "Infer the Org export backend from OUTPUT."
  (pcase (downcase (or (file-name-extension output) ""))
    ((or "adoc" "ascii" "txt") 'ascii)
    ((or "htm" "html") 'html)
    ((or "md" "markdown") 'md)
    ("odt" 'odt)
    ("pdf" 'pdf)
    ("tex" 'latex)
    (_ (error "Unsupported export extension for %s" output))))

(defun datamunge--export-pdf (output)
  "Export the current Org buffer to OUTPUT as PDF."
  (unless (executable-find "pdflatex")
    (error "pdflatex was not found in PATH"))
  (let ((pdf (org-latex-export-to-pdf nil nil nil nil nil)))
    (unless (and pdf (file-exists-p pdf))
      (error "Org did not produce a PDF"))
    (unless (file-equal-p pdf output)
      (copy-file pdf output t))
    output))

(defun datamunge--export-current-buffer (output)
  "Export the current Org buffer to OUTPUT."
  (let ((backend (datamunge--backend-for-output output)))
    (pcase backend
      ('pdf (datamunge--export-pdf output))
      (_ (org-export-to-file backend output nil nil nil nil nil)))))

(defun datamunge-org-export-file (input-org output-path)
  "Execute Babel for INPUT-ORG and export it to OUTPUT-PATH."
  (let ((input (expand-file-name input-org))
        (output (expand-file-name output-path)))
    (unless (file-exists-p input)
      (error "Input Org file not found: %s" input))
    (datamunge--ensure-parent-directory output)
    (datamunge--apply-direnv-environment (file-name-directory input))
    (let ((default-directory (file-name-directory input)))
      (with-current-buffer (find-file-noselect input)
        (unwind-protect
            (progn
              (org-mode)
              (datamunge--export-current-buffer output))
          (set-buffer-modified-p nil)
          (kill-buffer (current-buffer)))))))

(defun datamunge--print-usage-and-exit ()
  "Print usage for batch export and exit with a non-zero status."
  (princ
   (concat
    "Usage: emacs --batch -Q -l init.el -- <input.org> <output.{html,md,txt,tex,odt,pdf}>\n"
    "Example: emacs --batch -Q -l init.el -- examples/org/dense_linear_algebra.org build/org/dense_linear_algebra.html\n"))
  (kill-emacs 2))

(when noninteractive
  (let ((args (seq-filter (lambda (arg) (not (string= arg "--")))
                          command-line-args-left)))
    (cond
     ((null args) nil)
     ((= (length args) 2)
      (datamunge-org-export-file (nth 0 args) (nth 1 args))
      (princ (format "Wrote %s\n" (nth 1 args))))
     (t
      (datamunge--print-usage-and-exit)))))
